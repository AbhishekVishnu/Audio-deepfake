"""Evaluate the detector on out-of-distribution clips fetched from URLs.

Reads a CSV of `label,url` rows, downloads each via yt-dlp + ffmpeg, runs the
detector, and reports the same metrics evaluate.py produces. Cached downloads
in eval_cache/ so re-running is fast.

CSV format (no header required; `#` lines are ignored):

    real,https://www.youtube.com/watch?v=...
    fake,https://example.com/path/to/tts_demo.mp3
    real,https://archive.org/download/.../track.mp3

Usage:
  python evaluate_urls.py                       # uses test_urls.csv
  python evaluate_urls.py --csv my_urls.csv
  python evaluate_urls.py --threshold 0.4 --report-csv reports/url_results.csv

It will:
  - skip dead / unreachable URLs with a warning (won't crash the run)
  - cache fetched audio under eval_cache/ keyed by URL hash
  - print per-clip verdicts + the aggregate ablation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("evaluate_urls")
log.setLevel(logging.INFO)

BASE = Path(__file__).resolve().parent
CACHE_DIR = BASE / "eval_cache"
CACHE_DIR.mkdir(exist_ok=True)

import ingestion  # noqa: E402


@dataclass
class UrlResult:
    url: str
    label: int          # 1 = fake (positive class), 0 = real
    prob: float
    pred: int
    duration: float
    seconds: float
    error: str = ""


def parse_csv(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) != 2:
                log.warning("skipping malformed line: %s", line)
                continue
            label_str, url = parts[0].lower(), parts[1]
            if label_str in ("real", "0", "bonafide"):
                lbl = 0
            elif label_str in ("fake", "1", "spoof", "deepfake"):
                lbl = 1
            else:
                log.warning("skipping unknown label %r", label_str)
                continue
            rows.append((url, lbl))
    return rows


def cached_fetch(url: str) -> Path | None:
    """Return a path to a wav for `url`, fetching+caching if needed."""
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    cached = CACHE_DIR / f"{h}.wav"
    if cached.exists() and cached.stat().st_size > 1024:
        return cached
    try:
        wav = ingestion.fetch_audio(url, CACHE_DIR)
    except ingestion.IngestionError as exc:
        log.warning("FETCH FAILED %s — %s", url, exc)
        return None
    except Exception as exc:
        log.warning("FETCH FAILED %s — %s", url, exc)
        return None
    # Rename to deterministic cache key
    try:
        wav.rename(cached)
    except OSError:
        cached = wav
    return cached


def fmt_pct(x: float) -> str:
    return f"{x * 100:5.2f}%" if not np.isnan(x) else "  n/a "


def metrics(results: list[UrlResult], threshold: float) -> dict:
    valid = [r for r in results if not r.error]
    if not valid:
        return {"n": 0, "accuracy": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "f1": float("nan"), "eer": float("nan"),
                "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    y_true = np.array([r.label for r in valid])
    probs = np.array([r.prob for r in valid])
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    # EER sweep
    eer = float("nan")
    if len(np.unique(probs)) > 1:
        best = (1.0, 0.0)
        for t in np.unique(np.sort(probs)):
            p = (probs >= t).astype(int)
            fpr = ((p == 1) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())
            fnr = ((p == 0) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())
            d = abs(fpr - fnr)
            if d < best[0]:
                best = (d, (fpr + fnr) / 2)
        eer = best[1]
    return dict(n=len(y_true), accuracy=acc, precision=precision,
                recall=recall, f1=f1, eer=eer, tp=tp, tn=tn, fp=fp, fn=fn)


def print_report(name: str, m: dict, elapsed: float, n_failed: int) -> None:
    print()
    print(f"┌── {name} " + "─" * (52 - len(name)))
    print(f"│ samples           {m['n']}  (failed fetch: {n_failed})")
    print(f"│ accuracy          {fmt_pct(m['accuracy'])}")
    print(f"│ precision (fake)  {fmt_pct(m['precision'])}")
    print(f"│ recall    (fake)  {fmt_pct(m['recall'])}")
    print(f"│ F1               {fmt_pct(m['f1'])}")
    print(f"│ EER              {fmt_pct(m['eer'])}")
    print(f"│ confusion         TP={m['tp']:<3} FP={m['fp']:<3} TN={m['tn']:<3} FN={m['fn']:<3}")
    print(f"│ wall time         {elapsed:.1f}s")
    print(f"└" + "─" * 56)


def configure(mode: str) -> None:
    if mode == "hf":
        os.environ["DEEPFAKE_DISABLE_KERAS"] = "1"
        os.environ.pop("DEEPFAKE_KERAS_WEIGHT", None)
    elif mode == "keras":
        os.environ.pop("DEEPFAKE_DISABLE_KERAS", None)
        os.environ["DEEPFAKE_KERAS_WEIGHT"] = "1.0"
    elif mode == "ensemble":
        os.environ.pop("DEEPFAKE_DISABLE_KERAS", None)
        os.environ["DEEPFAKE_KERAS_WEIGHT"] = os.environ.get("DEEPFAKE_KERAS_WEIGHT", "0.5")


def reload_detector():
    import detector as detmod
    detmod._detector = None
    return detmod.get_detector()


def evaluate(detector, items: list[tuple[Path, str, int]], threshold: float,
             label: str) -> tuple[list[UrlResult], dict, float, int]:
    results: list[UrlResult] = []
    failed = 0
    t0 = time.time()
    for i, (wav, url, lbl) in enumerate(items, 1):
        try:
            t1 = time.time()
            pred = detector.predict(str(wav), with_saliency=False)
            results.append(UrlResult(
                url=url, label=lbl, prob=pred.fake_prob,
                pred=int(pred.fake_prob >= threshold),
                duration=pred.duration, seconds=time.time() - t1,
            ))
            true_str = "FAKE" if lbl else "REAL"
            pred_str = "FAKE" if results[-1].pred else "REAL"
            mark = "✓" if pred_str == true_str else "✗"
            short = url if len(url) < 60 else url[:57] + "…"
            print(f"  [{label}] {mark} {pred_str} ({pred.fake_prob:.3f}) ← {true_str}  {short}", flush=True)
        except Exception as exc:
            failed += 1
            results.append(UrlResult(url=url, label=lbl, prob=0, pred=0,
                                     duration=0, seconds=0, error=str(exc)))
            log.warning("INFER FAILED %s — %s", url, exc)
    return results, metrics(results, threshold), time.time() - t0, failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(BASE / "test_urls.csv"))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--ablate", action="store_true",
                    help="run hf-only, keras-only, ensemble back-to-back")
    ap.add_argument("--report-csv", default=None,
                    help="write per-clip predictions to a CSV report")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the eval_cache/ directory and refetch everything")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}\n"
                 f"Create it with `label,url` rows. See test_urls.csv for an example.")
    rows = parse_csv(csv_path)
    if not rows:
        sys.exit("No usable rows in CSV.")

    threshold = args.threshold if args.threshold is not None \
        else float(os.environ.get("DEEPFAKE_THRESHOLD", "0.5"))

    print(f"\n=== TrueVoice URL evaluation ===")
    print(f"csv:        {csv_path}")
    print(f"clips:      {sum(1 for _, l in rows if l == 0)} real · "
          f"{sum(1 for _, l in rows if l == 1)} fake")
    print(f"threshold:  {threshold}\n")

    # Stage 1: fetch all audio (with cache)
    print("Fetching clips...")
    items: list[tuple[Path, str, int]] = []
    fetch_failed = 0
    for url, lbl in rows:
        if args.no_cache:
            h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
            cache = CACHE_DIR / f"{h}.wav"
            if cache.exists():
                cache.unlink()
        wav = cached_fetch(url)
        if wav is None:
            fetch_failed += 1
            continue
        items.append((wav, url, lbl))
        print(f"  {'·' if (CACHE_DIR / wav.name).exists() else ' '} {url}")
    print(f"\n{len(items)}/{len(rows)} clips ready"
          f"{'  (' + str(fetch_failed) + ' fetch failures)' if fetch_failed else ''}")
    if not items:
        sys.exit("No clips fetched. Check your URLs.")

    # Stage 2: run modes
    modes = ["hf", "keras", "ensemble"] if args.ablate else ["ensemble"]
    summaries: dict[str, dict] = {}
    all_results: dict[str, list[UrlResult]] = {}

    for mode in modes:
        configure(mode)
        try:
            det = reload_detector()
        except Exception as exc:
            print(f"\n!! {mode} mode failed to initialise: {exc}")
            continue
        if mode == "keras" and det.keras_model is None:
            print(f"\n!! keras-only mode skipped — no Keras model loaded.")
            continue

        results, m, elapsed, n_inf_failed = evaluate(det, items, threshold, mode)
        print_report(mode, m, elapsed, fetch_failed + n_inf_failed)
        summaries[mode] = m
        all_results[mode] = results

    if args.ablate and len(summaries) > 1:
        print("\n┌── Ablation summary " + "─" * 36)
        print(f"│ {'mode':<10} {'accuracy':>10} {'F1':>10} {'EER':>10}")
        for mode, m in summaries.items():
            print(f"│ {mode:<10} {fmt_pct(m['accuracy']):>10} "
                  f"{fmt_pct(m['f1']):>10} {fmt_pct(m['eer']):>10}")
        print(f"└" + "─" * 56)

    if args.report_csv:
        out = Path(args.report_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mode", "true_label", "predicted", "fake_prob", "duration_s", "url", "error"])
            for mode, results in all_results.items():
                for r in results:
                    w.writerow([
                        mode,
                        "FAKE" if r.label else "REAL",
                        "FAKE" if r.pred else "REAL",
                        f"{r.prob:.4f}",
                        f"{r.duration:.2f}",
                        r.url,
                        r.error,
                    ])
        print(f"\nwrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
