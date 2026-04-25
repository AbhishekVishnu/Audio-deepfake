import csv
import io
import json
import logging
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

import augment
import history
import ingestion
from detector import Prediction, get_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("truevoice")

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "m4a", "ogg", "webm"}
MAX_UPLOAD_MB = 50

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
SPEC_DIR = BASE_DIR / "static" / "spec_cache"
FEEDBACK_DIR = BASE_DIR / "feedback"
UPLOAD_DIR.mkdir(exist_ok=True)
SPEC_DIR.mkdir(parents=True, exist_ok=True)
(FEEDBACK_DIR / "real").mkdir(parents=True, exist_ok=True)
(FEEDBACK_DIR / "fake").mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

history.init()


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _save_spectrogram(png_bytes: bytes) -> str:
    name = f"{uuid.uuid4().hex}.png"
    (SPEC_DIR / name).write_bytes(png_bytes)
    # Trim cache
    files = sorted(SPEC_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[100:]:
        try:
            old.unlink()
        except OSError:
            pass
    return f"/static/spec_cache/{name}"


def _serialize(p: Prediction, source: str, filename: str | None) -> dict:
    spec_url = _save_spectrogram(p.spectrogram_png)
    chunks = [{"start": round(c.start, 2), "end": round(c.end, 2),
               "fake_prob": round(c.fake_prob, 4)} for c in p.chunks]
    payload = {
        "label": p.label,
        "confidence": round(p.confidence, 2),
        "fake_prob": round(p.fake_prob, 4),
        "duration": round(p.duration, 2),
        "chunks": chunks,
        "saliency": [round(s, 4) for s in p.saliency],
        "spectrogram_url": spec_url,
        "models": p.models,
        "label_map": p.label_map,
        "aggregation": p.aggregation,
        "threshold": p.threshold,
        "source": source,
        "filename": filename,
    }
    history.save(
        source=source,
        filename=filename,
        label=p.label,
        confidence=p.confidence,
        fake_prob=p.fake_prob,
        duration=p.duration,
        chunks=chunks,
        models=p.models,
    )
    return payload


def _save_upload(file_storage) -> Path:
    safe = secure_filename(file_storage.filename or "upload")
    suffix = Path(safe).suffix or ".wav"
    saved = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    file_storage.save(saved)
    return saved


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="no file uploaded"), 400
    if not _allowed(file.filename):
        return jsonify(error=f"unsupported format; allowed: {sorted(ALLOWED_EXTENSIONS)}"), 400
    saved = _save_upload(file)
    try:
        result = get_detector().predict(str(saved))
        payload = _serialize(result, source="upload", filename=secure_filename(file.filename))
        # Keep file briefly so the client can run augment-predict against it
        payload["src_token"] = saved.name
        return jsonify(payload)
    except Exception as exc:
        log.exception("inference failed")
        return jsonify(error=f"inference failed: {exc}"), 500


@app.route("/api/predict-url", methods=["POST"])
def predict_url():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify(error="missing url"), 400
    try:
        wav = ingestion.fetch_audio(url, UPLOAD_DIR)
    except ingestion.IngestionError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        log.exception("ingestion failed")
        return jsonify(error=f"ingestion failed: {exc}"), 500
    try:
        result = get_detector().predict(str(wav))
        payload = _serialize(result, source="url", filename=url)
        payload["src_token"] = wav.name
        return jsonify(payload)
    except Exception as exc:
        log.exception("inference failed")
        return jsonify(error=f"inference failed: {exc}"), 500


@app.route("/api/augment-predict", methods=["POST"])
def augment_predict():
    """Re-run inference after applying an augmentation to the previously uploaded file."""
    data = request.get_json(silent=True) or {}
    token = data.get("src_token")
    kind = data.get("kind")
    if not token or kind not in augment.AUGMENT_KINDS:
        return jsonify(error="invalid request"), 400
    src = UPLOAD_DIR / token
    if not src.exists():
        return jsonify(error="source clip expired; re-upload"), 410
    try:
        aug_path = augment.apply(kind, str(src), UPLOAD_DIR)
        result = get_detector().predict(str(aug_path))
        payload = _serialize(result, source=f"augment:{kind}", filename=src.name)
        payload["augment_kind"] = kind
        try:
            aug_path.unlink()
        except OSError:
            pass
        return jsonify(payload)
    except Exception as exc:
        log.exception("augment inference failed")
        return jsonify(error=f"augment failed: {exc}"), 500


@app.route("/api/history")
def get_history():
    return jsonify(history.list_recent(limit=20))


@app.route("/api/feedback", methods=["POST"])
def feedback():
    """Save the analysed audio under feedback/{real,fake}/ for retraining.

    Body: {src_token, claimed_label: "real"|"fake"}
    """
    data = request.get_json(silent=True) or {}
    token = data.get("src_token")
    claimed = (data.get("claimed_label") or "").lower()
    if claimed not in ("real", "fake"):
        return jsonify(error="claimed_label must be 'real' or 'fake'"), 400
    if not token:
        return jsonify(error="missing src_token"), 400
    src = UPLOAD_DIR / token
    if not src.exists():
        return jsonify(error="source clip expired; re-upload"), 410

    dest_dir = FEEDBACK_DIR / claimed
    dest = dest_dir / f"{uuid.uuid4().hex}{Path(token).suffix or '.wav'}"
    try:
        dest.write_bytes(src.read_bytes())
    except OSError as exc:
        return jsonify(error=f"could not save feedback: {exc}"), 500

    counts = {
        "real": sum(1 for _ in (FEEDBACK_DIR / "real").iterdir()),
        "fake": sum(1 for _ in (FEEDBACK_DIR / "fake").iterdir()),
    }
    log.info("feedback saved → %s (%s); counts=%s", dest.name, claimed, counts)
    return jsonify(saved=str(dest.relative_to(BASE_DIR)), counts=counts)


@app.route("/api/feedback/stats")
def feedback_stats():
    counts = {
        "real": sum(1 for _ in (FEEDBACK_DIR / "real").iterdir()) if (FEEDBACK_DIR / "real").exists() else 0,
        "fake": sum(1 for _ in (FEEDBACK_DIR / "fake").iterdir()) if (FEEDBACK_DIR / "fake").exists() else 0,
    }
    return jsonify(counts=counts, total=counts["real"] + counts["fake"])


@app.route("/api/export/<fmt>", methods=["POST"])
def export(fmt: str):
    if fmt not in {"json", "csv"}:
        abort(404)
    data = request.get_json(silent=True) or {}
    chunks = data.get("chunks") or []
    label = data.get("label", "")
    confidence = data.get("confidence", 0)
    fake_prob = data.get("fake_prob", 0)
    filename = data.get("filename", "analysis")

    if fmt == "json":
        buf = io.BytesIO(json.dumps(data, indent=2).encode("utf-8"))
        return send_file(buf, mimetype="application/json", as_attachment=True,
                         download_name=f"{filename}_analysis.json")

    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["filename", filename])
    w.writerow(["label", label])
    w.writerow(["confidence", confidence])
    w.writerow(["overall_fake_prob", fake_prob])
    w.writerow([])
    w.writerow(["chunk_start_s", "chunk_end_s", "fake_prob"])
    for c in chunks:
        w.writerow([c.get("start"), c.get("end"), c.get("fake_prob")])
    buf = io.BytesIO(out.getvalue().encode("utf-8"))
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name=f"{filename}_analysis.csv")


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"file too large; max {MAX_UPLOAD_MB} MB"), 413


if __name__ == "__main__":
    import os
    log.info("warming up model(s)...")
    get_detector()
    port = int(os.environ.get("PORT", "5001"))  # 5000 collides with macOS AirPlay
    log.info("ready — http://127.0.0.1:%d", port)
    app.run(host="127.0.0.1", port=port, debug=False)
