"""Evaluate detector quality on the local dataset.

Walks dataset/real and dataset/fake, runs the configured detector on a sample
of clips from each class, and reports accuracy / precision / recall / F1 / EER
plus a confusion matrix and a CSV of per-clip predictions for further analysis.

Usage:
  python evaluate.py                      # default: 50 clips per class
  python evaluate.py --num 100            # 100 per class (200 total)
  python evaluate.py --num 25 --ablate    # run hf-only / keras-only / ensemble
  python evaluate.py --num 50 --csv out/results.csv
  python evaluate.py --num 50 --threshold 0.4

Tips:
  - First run downloads the HF model (~370 MB).
  - Use a small --num while iterating; bump it when reporting final numbers.
  - --ablate is what generates the table that fills two pages of a project report.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Quiet TF/tokenizer noise before they import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger("evaluate")
log.setLevel(logging.INFO)

BASE = Path(__file__).resolve().parent
REAL_DIR = BASE / "dataset" / "real"
FAKE_DIR = BASE / "dataset" / "fake"


@dataclass
class ClipResult:
    path: str
    label: int          # 1 = fake (positive class), 0 = real
    prob: float         # predicted fake probability
    pred: int           # 1 if prob >= threshold
    duration: float
    seconds: float      # wall-clock seconds spent


def collect(num: int, seed: int) -> tuple[list[Path], list[Path]]:
    if not REAL_DIR.exists() or not FAKE_DIR.exists():
        sys.exit(f"Dataset not found. Expected {REAL_DIR} and {FAKE_DIR}.")
    rng = random.Random(seed)
    real = sorted(p for p in REAL_DIR.iterdir() if p.suffix.lower() == ".wav")
    fake = sorted(p for p in FAKE_DIR.iterdir() if p.suffix.lower() == ".wav")
    if not real or not fake:
        sys.exit("No .wav files found in dataset/real or dataset/fake.")
    rng.shuffle(real)
    rng.shuffle(fake)
    return real[:num], fake[:num]


def metrics(results: list[ClipResult], threshold: float) -> dict:
    """Standard binary classification metrics + EER."""
    y_true = np.array([r.label for r in results])
    probs = np.array([r.prob for r in results])
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    # Equal Error Rate (EER): the threshold where FPR == FNR
    eer = compute_eer(y_true, probs)

    return {
        "n": len(y_true),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "eer": eer,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def compute_eer(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Sweep thresholds; find where FPR == FNR (the equal error rate)."""
    sorted_probs = np.unique(np.sort(probs))
    if len(sorted_probs) < 2:
        return float("nan")
    best = (1.0, 0.0)
    for t in sorted_probs:
        preds = (probs >= t).astype(int)
        fpr = ((preds == 1) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())
        fnr = ((preds == 0) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())
        diff = abs(fpr - fnr)
        if diff < best[0]:
            best = (diff, (fpr + fnr) / 2)
    return best[1]


def fmt_pct(x: float) -> str:
    return f"{x * 100:5.2f}%" if not np.isnan(x) else "  n/a "


def print_report(name: str, m: dict, elapsed: float) -> None:
    print()
    print(f"┌── {name} " + "─" * (52 - len(name)))
    print(f"│ samples           {m['n']}")
    print(f"│ accuracy          {fmt_pct(m['accuracy'])}")
    print(f"│ precision (fake)  {fmt_pct(m['precision'])}")
    print(f"│ recall    (fake)  {fmt_pct(m['recall'])}")
    print(f"│ F1               {fmt_pct(m['f1'])}")
    print(f"│ EER              {fmt_pct(m['eer'])}")
    print(f"│ confusion         TP={m['tp']:<4} FP={m['fp']:<4} TN={m['tn']:<4} FN={m['fn']:<4}")
    print(f"│ wall time         {elapsed:.1f}s ({elapsed / max(1, m['n']):.2f}s/clip)")
    print(f"└" + "─" * 56)


def evaluate(detector, clips: list[tuple[Path, int]], threshold: float,
             label: str) -> tuple[list[ClipResult], dict, float]:
    results: list[ClipResult] = []
    t0 = time.time()
    for i, (path, lbl) in enumerate(clips, 1):
        try:
            t1 = time.time()
            pred = detector.predict(str(path), with_saliency=False)
            results.append(ClipResult(
                path=str(path),
                label=lbl,
                prob=pred.fake_prob,
                pred=int(pred.fake_prob >= threshold),
                duration=pred.duration,
                seconds=time.time() - t1,
            ))
            if i % 5 == 0 or i == len(clips):
                rolling = sum(1 for r in results if r.pred == r.label) / len(results)
                print(f"  [{label}] {i}/{len(clips)}  rolling-acc={rolling*100:5.2f}%", flush=True)
        except Exception as exc:
            log.warning("FAILED %s — %s", path.name, exc)
    elapsed = time.time() - t0
    m = metrics(results, threshold)
    return results, m, elapsed


def write_csv(path: Path, results: list[ClipResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_label", "fake_prob", "predicted", "duration_s", "infer_s"])
        for r in results:
            w.writerow([r.path, "FAKE" if r.label else "REAL", f"{r.prob:.4f}",
                        "FAKE" if r.pred else "REAL", f"{r.duration:.2f}",
                        f"{r.seconds:.2f}"])


def configure(mode: str) -> None:
    """Set env vars to switch the detector between hf-only / keras-only / ensemble."""
    if mode == "hf":
        os.environ["DEEPFAKE_DISABLE_KERAS"] = "1"
        os.environ.pop("DEEPFAKE_KERAS_WEIGHT", None)
    elif mode == "keras":
        os.environ.pop("DEEPFAKE_DISABLE_KERAS", None)
        os.environ["DEEPFAKE_KERAS_WEIGHT"] = "1.0"   # 100% keras
    elif mode == "ensemble":
        os.environ.pop("DEEPFAKE_DISABLE_KERAS", None)
        os.environ["DEEPFAKE_KERAS_WEIGHT"] = os.environ.get("DEEPFAKE_KERAS_WEIGHT", "0.5")


def reload_detector():
    """Force re-init of the singleton detector after env var changes."""
    import detector as detmod
    detmod._detector = None
    return detmod.get_detector()


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate the deepfake detector on the bundled dataset.")
    ap.add_argument("--num", type=int, default=50, help="clips per class (default 50)")
    ap.add_argument("--threshold", type=float, default=None, help="override decision threshold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", type=str, default=None, help="path to write per-clip CSV")
    ap.add_argument("--ablate", action="store_true",
                    help="run hf-only, keras-only, and ensemble back-to-back")
    args = ap.parse_args()

    threshold = args.threshold if args.threshold is not None \
        else float(os.environ.get("DEEPFAKE_THRESHOLD", "0.5"))

    print(f"\n=== TrueVoice evaluation ===")
    print(f"sampling {args.num} clips per class from {REAL_DIR.name}/, {FAKE_DIR.name}/")
    print(f"threshold = {threshold}")

    real, fake = collect(args.num, args.seed)
    clips = [(p, 0) for p in real] + [(p, 1) for p in fake]
    random.Random(args.seed).shuffle(clips)
    print(f"loaded {len(clips)} clips total\n")

    modes = ["hf", "keras", "ensemble"] if args.ablate else ["ensemble"]
    summaries = {}

    for mode in modes:
        configure(mode)
        try:
            det = reload_detector()
        except Exception as exc:
            print(f"\n!! {mode} mode failed to initialise: {exc}")
            continue

        # If keras-only mode but no keras model loaded, skip
        if mode == "keras" and det.keras_model is None:
            print(f"\n!! keras-only mode skipped — no Keras model loaded.")
            continue

        results, m, elapsed = evaluate(det, clips, threshold, mode)
        print_report(mode, m, elapsed)
        summaries[mode] = m

        if args.csv:
            csv_path = Path(args.csv)
            if args.ablate:
                csv_path = csv_path.with_stem(f"{csv_path.stem}_{mode}")
            write_csv(csv_path, results)
            print(f"  wrote {csv_path}")

    if args.ablate and len(summaries) > 1:
        print("\n┌── Ablation summary " + "─" * 36)
        print(f"│ {'mode':<10} {'accuracy':>10} {'F1':>10} {'EER':>10}")
        for mode, m in summaries.items():
            print(f"│ {mode:<10} {fmt_pct(m['accuracy']):>10} {fmt_pct(m['f1']):>10} {fmt_pct(m['eer']):>10}")
        print(f"└" + "─" * 56)

    return 0


if __name__ == "__main__":
    sys.exit(main())
