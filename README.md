# TrueVoice — Audio Deepfake Detection

Forensic audio analysis console. Detects AI-generated speech with a Wav2Vec2 ensemble plus a locally-trained fusion CNN, with per-chunk timeline analysis, mel-spectrogram visualisation, robustness probing, live recording, URL ingestion, and a human-in-the-loop feedback queue for retraining.

> **For research and educational use only.**

## Features

- **Multi-input intake**: file upload (drag-drop / paste / click), in-browser mic recording, YouTube/URL ingestion via `yt-dlp`, side-by-side A/B comparison.
- **Two-model ensemble**: HuggingFace Wav2Vec2 detector + your locally-trained Keras fusion CNN, weighted blend.
- **Chunked inference**: 5-second overlapping windows, top-K mean aggregation (catches localised fakery), saliency from input gradients.
- **Visualisation**: per-chunk fake-probability timeline, mel-spectrogram with per-chunk verdict overlay, animated confidence gauge.
- **Robustness panel**: one-click apply noise / telephone-band / pitch / reverb / speed perturbation, see Δpp from baseline.
- **Feedback loop**: flag mis-classified clips → saved to `feedback/{real,fake}/` → fine-tune with `train.py`.
- **History**: SQLite-backed last 20 analyses.
- **Export**: per-chunk JSON / CSV download.
- **Evaluation harnesses**: `evaluate.py` for in-distribution metrics on `dataset/`, `evaluate_urls.py` for OOD URL-based testing with ablation across HF / Keras / ensemble.

## Quick start

```bash
# 1. Clone
git clone https://github.com/AbhishekVishnu/Audio-deepfake.git
cd Audio-deepfake

# 2. venv + deps
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. ffmpeg (only needed for the URL tab)
brew install ffmpeg                       # macOS
# or: winget install Gyan.FFmpeg          # Windows
# or: sudo apt install ffmpeg             # Linux

# 4. Run
python app.py                              # → http://127.0.0.1:5001
```

The first prediction downloads the HuggingFace model (~370 MB) into the local cache; subsequent runs are fast.

## Configuration (env vars)

```
DEEPFAKE_MODEL                primary HF model id
                              default: motheecreator/Deepfake-audio-detection
DEEPFAKE_MODEL_2              optional second HF model (HF×HF ensemble)
DEEPFAKE_KERAS_MODEL          path to .keras (default: model/fusion_model_final.keras)
DEEPFAKE_DISABLE_KERAS=1      force HF-only
DEEPFAKE_KERAS_WEIGHT=0.5     ensemble blend (0 = HF-only, 1 = Keras-only)
DEEPFAKE_THRESHOLD=0.5        decision threshold (lower = more sensitive)
DEEPFAKE_AGGREGATION          mean | max | topk_mean | p90  (default: topk_mean)
DEEPFAKE_TOPK=3               K for topk_mean
DEEPFAKE_FAKE_INDEX           override label inversion (e.g. "0")
PORT=5001                     Flask port (5000 collides with macOS AirPlay)
INGEST_MAX_DURATION_S=1800    yt-dlp duration cap, seconds
```

## Layout

```
app.py              Flask app + JSON API
detector.py         Inference engine (HF + Keras ensemble)
ingestion.py        yt-dlp wrapper for URL fetching
augment.py          Audio perturbations
history.py          SQLite history
train.py            Fine-tune the Keras model on dataset/ + feedback/
evaluate.py         In-distribution evaluation
evaluate_urls.py    Out-of-distribution evaluation
build_guide_pdf.py  Render the presentation guide to PDF

templates/          UI (single-page)
static/css/         Editorial-forensic theme
static/js/          Frontend logic (drag-drop, WaveSurfer, charts, feedback)

model/              Trained Keras fusion model (.keras)
test_urls.csv       Curated OOD URLs for evaluate_urls.py

PRESENTATION_GUIDE.md      Walkthrough of changes, demo flow, viva prep
PRESENTATION_GUIDE.pdf     Same, print-ready
```

Not in the repo (excluded by `.gitignore`):
`dataset/`, `output/`, `feedback/`, `uploads/`, `eval_cache/`, `static/spec_cache/`, `history.db`, `.venv/` — see the original `Audio deepfake.rar` archive for the dataset.

## Running the evaluations

```bash
# In-distribution ablation across HF / Keras / Ensemble
python evaluate.py --num 60 --ablate

# Out-of-distribution test on URLs (populate test_urls.csv first)
python evaluate_urls.py --ablate

# Fine-tune the offline model on accumulated feedback
python train.py --epochs 5
```

## Documentation

- **`PRESENTATION_GUIDE.md`** / **`PRESENTATION_GUIDE.pdf`** — full handover doc: Windows setup, what changed and why, concepts, demo flow, likely viva questions.
- **`CLAUDE.md`** — terse system overview for AI assistants working on the codebase.

## Acknowledgements

- Wav2Vec2 detector: `motheecreator/Deepfake-audio-detection` on Hugging Face
- WaveSurfer.js for the in-browser waveform
- yt-dlp + ffmpeg for URL ingestion
