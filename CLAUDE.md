# TrueVoice — Audio Deepfake Detection

Flask app that classifies an audio clip as REAL or FAKE using a pretrained Wav2Vec2-based deepfake detector. Supports file upload, mic recording, YouTube/URL ingestion, side-by-side comparison, per-chunk timeline analysis, mel-spectrogram visualisation with per-chunk overlay, robustness testing under perturbation, history persistence, and JSON/CSV export.

## Run

```bash
# 1. Fresh venv (the bundled Windows .venv has been removed)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Python deps
pip install -r requirements.txt

# 3. Install ffmpeg for URL ingestion (one-time)
brew install ffmpeg

# 4. Start
python app.py        # → http://127.0.0.1:5000
```

First run downloads ~370 MB of model weights from HuggingFace (cached after).

### Environment variables

- `DEEPFAKE_MODEL` — primary HF model id. Default: `MelodyMachine/Deepfake-audio-detection-V2`
- `DEEPFAKE_MODEL_2` — optional ensemble partner; if set, predictions are averaged.

Apple Silicon (M-series) auto-uses MPS device; CUDA used if available; otherwise CPU.

## Architecture

```
app.py            Flask app + JSON API endpoints
detector.py       Inference: load model(s), chunked predict, saliency, spectrogram render
ingestion.py      yt-dlp wrapper for YouTube/URL → wav
augment.py        Audio perturbations (noise, telephone-band, pitch, reverb, speed)
history.py        SQLite persistence of last 50 analyses
templates/index.html   Single-page UI
static/css/app.css     Custom styling (no Bootstrap)
static/js/app.js       Tabs, drag-drop, mic recording, charts, history drawer
static/spec_cache/     Generated mel-spectrograms (auto-pruned, last 100)
history.db        SQLite (auto-created)
uploads/          Transient upload buffer
```

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | UI |
| `/api/predict` | POST multipart `file` | Analyse uploaded audio |
| `/api/predict-url` | POST JSON `{url}` | Fetch + analyse remote audio (yt-dlp) |
| `/api/augment-predict` | POST JSON `{src_token, kind}` | Re-analyse after applying perturbation |
| `/api/history` | GET | Last 20 analyses |
| `/api/export/json\|csv` | POST JSON | Download current result |

`src_token` returned by `/api/predict` and `/api/predict-url` is a short-lived handle used by augmentation to avoid re-uploading.

## Detection pipeline

1. Load audio via librosa, mono, resample to 16 kHz.
2. Split into overlapping 5-second windows (50% overlap).
3. For each model in the ensemble: feature-extract → forward → softmax → take fake-class prob.
4. Average across models and chunks.
5. Compute per-chunk gradient saliency (magnitude of ∂fake_logit/∂input averaged over each window).
6. Render mel-spectrogram (128 mels, n_fft 2048, hop 512) to PNG via numpy + PIL with a built-in viridis LUT (no matplotlib dep).

## Frontend features

- Four input modes: Upload (drag/drop/paste), Record (MediaRecorder), URL, Compare two clips.
- Animated radial gauge + per-class probability bars.
- Per-chunk timeline canvas: line + area chart of fake_prob with 0.5 threshold and saliency strip.
- Mel-spectrogram with translucent per-chunk verdict overlay (red = fake, green = real, intensity = confidence).
- Robustness panel: one-click apply noise/lowband/pitch/reverb/speed → re-runs inference and shows Δ pp from baseline.
- History drawer (last 20) with inline verdicts.
- Export current analysis as JSON or CSV.

## Legacy — for reference, not used

- `model/fusion_model_final.keras` — old fusion CNN. Replaced by HF pretrained model.
- `preprocess_audio.py` — old offline preprocessing for the legacy model. Not invoked by app.
- `dataset/`, `output/` — legacy training data and intermediates. Not required at inference time.

These can be deleted once you're confident the new pipeline works.

## Known caveats

- yt-dlp + ffmpeg are runtime requirements only for the URL tab.
- First request after server start triggers model download → can take 30–90 s.
- `temp_wav` files in `uploads/` are kept briefly so augmentation can re-analyse the same source; they are cleaned only on app restart. For production, add a TTL job.
- Saliency is gradient-based and time-only; not a true Grad-CAM heatmap. Per-chunk overlay on the spectrogram conveys time-localised verdicts.
