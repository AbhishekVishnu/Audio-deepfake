"""Deepfake audio detector using one or two pretrained Wav2Vec2-style classifiers.

Returns per-chunk probabilities, a rendered mel-spectrogram (PNG bytes),
and a per-chunk saliency curve. Supports optional ensemble via env var.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field

import librosa
import numpy as np
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

log = logging.getLogger(__name__)

# Path to the locally-trained Keras fusion model (default: bundled).
_DEFAULT_KERAS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "model", "fusion_model_final.keras")

# All other config is read inside Detector.__init__ so evaluate.py can flip
# env vars between modes without re-importing the module.

TARGET_SR = 16000
CHUNK_SECONDS = 5.0
CHUNK_OVERLAP = 0.5

REAL_KEYWORDS = ("real", "bonafide", "bona-fide", "genuine", "human", "authentic")
FAKE_KEYWORDS = ("fake", "spoof", "synthetic", "deepfake", "ai-gen", "generated", "tts", "vc", "cloned")


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ChunkResult:
    start: float
    end: float
    fake_prob: float


@dataclass
class Prediction:
    label: str
    confidence: float
    fake_prob: float
    chunks: list[ChunkResult]
    duration: float
    saliency: list[float]              # per-chunk saliency, 0–1
    spectrogram_png: bytes             # mel-spec as PNG
    models: list[str] = field(default_factory=list)
    label_map: list[dict] = field(default_factory=list)  # diagnostics: {model, labels, fake_idx}
    aggregation: str = "topk_mean"
    threshold: float = 0.5


class _SingleModel:
    def __init__(self, model_id: str, device: torch.device, override_idx: int | None = None) -> None:
        self.model_id = model_id
        self.device = device
        log.info("Loading %s on %s", model_id, device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        self.model.to(device).eval()
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.fake_idx = override_idx if override_idx is not None else self._resolve_fake_index()
        log.info("  labels: %s", self.id2label)
        log.info("  fake_idx resolved to: %d (%s)", self.fake_idx,
                 self.id2label.get(self.fake_idx, "?"))

    def _resolve_fake_index(self) -> int:
        labels = {idx: name.lower() for idx, name in self.id2label.items()}

        # Pass 1: explicit fake/spoof keyword
        for idx, name in labels.items():
            if any(t in name for t in FAKE_KEYWORDS):
                return idx

        # Pass 2: explicit real/bonafide keyword → take the OTHER index
        for idx, name in labels.items():
            if any(t in name for t in REAL_KEYWORDS):
                others = [i for i in labels if i != idx]
                if others:
                    return others[0]

        # Pass 3: ASVspoof convention — index 1 is "spoof", 0 is "bonafide"
        # Most LABEL_0/LABEL_1 binary classifiers follow this.
        if len(labels) == 2:
            log.warning("Could not resolve fake index from labels %s — defaulting to 1. "
                        "Override with DEEPFAKE_FAKE_INDEX env var if wrong.", labels)
            return 1

        # Multiclass: default to 0 (placeholder)
        log.warning("Multiclass model with no obvious fake label; using index 0. Override with DEEPFAKE_FAKE_INDEX.")
        return 0

    @torch.no_grad()
    def fake_probs(self, chunks: list[np.ndarray]) -> np.ndarray:
        inputs = self.feature_extractor(
            chunks, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs[:, self.fake_idx]


class _KerasModel:
    """Adapter for the locally-trained fusion CNN.

    Expects two inputs: mel-spectrogram (1, 128, 94, 1) and raw waveform (1, 48000, 1)
    derived from a 3-second 16 kHz mono clip. Output is a sigmoid prob of FAKE.
    """

    KERAS_DURATION_S = 3
    KERAS_SAMPLES = 48000  # 3 s * 16 kHz

    def __init__(self, path: str) -> None:
        from pathlib import Path
        # Lazy-import tensorflow so HF-only deployments don't pay the import cost.
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.models import load_model

        self.model_id = f"keras:{Path(path).name}"
        log.info("Loading %s", self.model_id)
        self.model = load_model(path, compile=False)
        # Convention: sigmoid output, prob > 0.5 ⇒ FAKE. We expose this so the
        # diagnostics panel can show it alongside the HF model's resolved labels.
        self.id2label = {0: "REAL", 1: "FAKE"}
        self.fake_idx = 1
        log.info("  fusion model loaded — input shapes: %s",
                 [tuple(i.shape) for i in self.model.inputs])

    def fake_probs(self, chunks: list[np.ndarray]) -> np.ndarray:
        specs = []
        waves = []
        for c in chunks:
            # Trim each 5-s ensemble chunk down to the 3-s window the Keras model
            # was trained on. Use the centre of the chunk where speech is most stable.
            n = self.KERAS_SAMPLES
            if len(c) >= n:
                start = (len(c) - n) // 2
                window = c[start:start + n]
            else:
                window = np.pad(c, (0, n - len(c)))
            mel = librosa.feature.melspectrogram(
                y=window, sr=TARGET_SR, n_fft=2048, hop_length=512, n_mels=128
            )
            mel_db = librosa.power_to_db(mel, ref=np.max).astype("float32")
            # Match training shape (128, 94). Trim/pad along the time axis defensively.
            if mel_db.shape[1] >= 94:
                mel_db = mel_db[:, :94]
            else:
                mel_db = np.pad(mel_db, ((0, 0), (0, 94 - mel_db.shape[1])))
            specs.append(mel_db)
            waves.append(window.astype("float32"))

        spec_batch = np.stack(specs)[..., np.newaxis]   # (B, 128, 94, 1)
        wave_batch = np.stack(waves)[..., np.newaxis]   # (B, 48000, 1)
        preds = self.model.predict([spec_batch, wave_batch], verbose=0)
        # The trained model outputs a sigmoid scalar per sample; squeeze to 1-D.
        return np.asarray(preds, dtype=np.float32).reshape(-1)


def _parse_index_override(spec: str | None) -> dict[str, int]:
    """Parse DEEPFAKE_FAKE_INDEX. Forms: '1' (all models) or 'modelA:1,modelB:0'."""
    if not spec:
        return {}
    spec = spec.strip()
    if ":" not in spec:
        return {"_all": int(spec)}
    out: dict[str, int] = {}
    for token in spec.split(","):
        if ":" in token:
            k, v = token.split(":", 1)
            out[k.strip()] = int(v.strip())
    return out


class Detector:
    def __init__(self) -> None:
        self.device = _select_device()

        # Read all knobs at construction time (not import time) so that
        # evaluate.py can flip env vars between modes between detector reloads.
        primary_model = os.environ.get("DEEPFAKE_MODEL", "motheecreator/Deepfake-audio-detection")
        secondary_model = os.environ.get("DEEPFAKE_MODEL_2")
        keras_path = os.environ.get("DEEPFAKE_KERAS_MODEL", _DEFAULT_KERAS)
        disable_keras = os.environ.get("DEEPFAKE_DISABLE_KERAS", "").lower() in ("1", "true", "yes")

        self.keras_weight = float(os.environ.get("DEEPFAKE_KERAS_WEIGHT", "0.5"))
        self.threshold = float(os.environ.get("DEEPFAKE_THRESHOLD", "0.5"))
        self.aggregation = os.environ.get("DEEPFAKE_AGGREGATION", "topk_mean").lower()
        self.topk = int(os.environ.get("DEEPFAKE_TOPK", "3"))

        log.info("=" * 60)
        log.info("TrueVoice detector initialising on %s", self.device)
        log.info("Aggregation: %s (topk=%d) · Threshold: %.2f",
                 self.aggregation, self.topk, self.threshold)
        log.info("Keras: disabled=%s · weight=%.2f · path=%s",
                 disable_keras, self.keras_weight, keras_path)

        overrides = _parse_index_override(os.environ.get("DEEPFAKE_FAKE_INDEX"))
        if overrides:
            log.info("Fake-index overrides from env: %s", overrides)

        def _override_for(model_id: str) -> int | None:
            return overrides.get(model_id, overrides.get("_all"))

        # HF transformer model(s) — index 0 is the primary, used for saliency.
        self.hf_models: list[_SingleModel] = [
            _SingleModel(primary_model, self.device, override_idx=_override_for(primary_model))
        ]
        if secondary_model:
            try:
                self.hf_models.append(_SingleModel(
                    secondary_model, self.device, override_idx=_override_for(secondary_model)
                ))
            except Exception as exc:
                log.warning("secondary model %s failed to load: %s", secondary_model, exc)

        # Optional Keras fusion model (the user's locally-trained CNN).
        self.keras_model: _KerasModel | None = None
        if not disable_keras and os.path.exists(keras_path) and os.path.getsize(keras_path) > 1024:
            try:
                self.keras_model = _KerasModel(keras_path)
                log.info("Keras model joined ensemble · weight=%.2f", self.keras_weight)
            except Exception as exc:
                log.warning("Keras model at %s failed to load: %s", keras_path, exc)
        elif disable_keras:
            log.info("Keras model disabled by DEEPFAKE_DISABLE_KERAS")
        else:
            log.info("Keras model not loaded (path missing or empty): %s", keras_path)
        log.info("=" * 60)

    @property
    def models(self) -> list:
        """All models for diagnostics. HF models first, then Keras."""
        out = list(self.hf_models)
        if self.keras_model is not None:
            out.append(self.keras_model)
        return out

    @property
    def model_ids(self) -> list[str]:
        return [m.model_id for m in self.models]

    def _load(self, path: str) -> np.ndarray:
        audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
        return audio.astype(np.float32)

    def _chunk(self, audio: np.ndarray) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
        chunk_len = int(CHUNK_SECONDS * TARGET_SR)
        if len(audio) <= chunk_len:
            padded = np.pad(audio, (0, max(0, chunk_len - len(audio))))
            return [padded], [(0.0, len(audio) / TARGET_SR)]

        step = int(chunk_len * (1 - CHUNK_OVERLAP))
        chunks: list[np.ndarray] = []
        spans: list[tuple[float, float]] = []
        for start in range(0, len(audio) - chunk_len + 1, step):
            chunks.append(audio[start:start + chunk_len])
            spans.append((start / TARGET_SR, (start + chunk_len) / TARGET_SR))
        if (len(audio) - chunk_len) % step != 0:
            chunks.append(audio[-chunk_len:])
            spans.append(((len(audio) - chunk_len) / TARGET_SR, len(audio) / TARGET_SR))
        return chunks, spans

    def _render_spectrogram(self, audio: np.ndarray) -> bytes:
        n_fft = 2048
        hop = 512
        n_mels = 128
        mel = librosa.feature.melspectrogram(
            y=audio, sr=TARGET_SR, n_fft=n_fft, hop_length=hop, n_mels=n_mels
        )
        db = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T), values <= 0
        # Normalize to 0–1, flip so high freqs at top
        norm = (db - db.min()) / (db.max() - db.min() + 1e-9)
        norm = np.flipud(norm)
        # Apply a perceptual gamma + viridis-like colormap
        idx = np.clip((norm * 255).astype(np.uint8), 0, 255)
        cmap = _viridis_lut()
        rgb = cmap[idx]  # (n_mels, T, 3)
        img = Image.fromarray(rgb, mode="RGB")
        # Resize to a wide aspect for the UI
        target_w = max(800, min(1600, rgb.shape[1] * 2))
        target_h = 256
        img = img.resize((target_w, target_h), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def _saliency(self, audio: np.ndarray, spans: list[tuple[float, float]]) -> list[float]:
        """Per-chunk saliency from input gradient on the primary HF model."""
        m = self.hf_models[0]
        chunk_len = int(CHUNK_SECONDS * TARGET_SR)
        scores: list[float] = []
        for s, e in spans:
            start = int(s * TARGET_SR)
            window = audio[start:start + chunk_len]
            if len(window) < chunk_len:
                window = np.pad(window, (0, chunk_len - len(window)))
            wave = torch.tensor(window, dtype=torch.float32, device=m.device).unsqueeze(0)
            wave.requires_grad_(True)
            inputs = m.feature_extractor(
                [window], sampling_rate=TARGET_SR, return_tensors="pt", padding=True
            )
            input_values = inputs["input_values"].to(m.device).requires_grad_(True)
            logits = m.model(input_values=input_values).logits
            target = logits[0, m.fake_idx]
            grad = torch.autograd.grad(target, input_values)[0]
            score = float(grad.abs().mean().detach().cpu())
            scores.append(score)
        if not scores:
            return []
        arr = np.array(scores)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        else:
            arr = np.zeros_like(arr)
        return arr.tolist()

    def predict(self, path: str, with_saliency: bool = True) -> Prediction:
        audio = self._load(path)
        if audio.size == 0:
            raise ValueError("Empty audio")
        duration = len(audio) / TARGET_SR

        chunks, spans = self._chunk(audio)

        # HF models first — straight average across however many are loaded
        hf_probs_list = [m.fake_probs(chunks) for m in self.hf_models]
        hf_avg = np.mean(np.stack(hf_probs_list, axis=0), axis=0)

        # If the trained Keras fusion model is available, blend it in by weight.
        if self.keras_model is not None:
            keras_probs = self.keras_model.fake_probs(chunks)
            ens = (1 - self.keras_weight) * hf_avg + self.keras_weight * keras_probs
            log.info("  hf_avg=%s  keras=%s  blended=%s",
                     [f"{p:.3f}" for p in hf_avg],
                     [f"{p:.3f}" for p in keras_probs],
                     [f"{p:.3f}" for p in ens])
        else:
            ens = hf_avg

        chunk_results = [ChunkResult(s, e, float(p)) for (s, e), p in zip(spans, ens)]
        fake_prob = self._aggregate(ens)

        log.info("Predict: chunks=%d  per-chunk=%s  agg=%s → %.4f",
                 len(ens), [f"{p:.3f}" for p in ens], self.aggregation, fake_prob)

        if fake_prob >= self.threshold:
            label, confidence = "FAKE", fake_prob * 100
        else:
            label, confidence = "REAL", (1 - fake_prob) * 100

        spec_png = self._render_spectrogram(audio)
        saliency = self._saliency(audio, spans) if with_saliency else []

        label_map = [
            {"model": m.model_id, "labels": m.id2label, "fake_idx": m.fake_idx}
            for m in self.models
        ]

        return Prediction(
            label=label,
            confidence=confidence,
            fake_prob=fake_prob,
            chunks=chunk_results,
            duration=duration,
            saliency=saliency,
            spectrogram_png=spec_png,
            models=self.model_ids,
            label_map=label_map,
            aggregation=self.aggregation,
            threshold=self.threshold,
        )

    def _aggregate(self, probs: np.ndarray) -> float:
        if probs.size == 0:
            return 0.0
        if self.aggregation == "mean":
            return float(probs.mean())
        if self.aggregation == "max":
            return float(probs.max())
        if self.aggregation == "p90":
            return float(np.percentile(probs, 90))
        # topk_mean (default): average of the K most-suspicious chunks
        k = min(max(1, self.topk), len(probs))
        return float(np.sort(probs)[-k:].mean())


_VIRIDIS: np.ndarray | None = None


def _viridis_lut() -> np.ndarray:
    """Tiny viridis-like 256-entry RGB LUT (no matplotlib dependency)."""
    global _VIRIDIS
    if _VIRIDIS is not None:
        return _VIRIDIS
    # Anchors approximating viridis
    anchors = np.array([
        [68, 1, 84],
        [72, 35, 116],
        [64, 67, 135],
        [52, 94, 141],
        [41, 120, 142],
        [32, 144, 140],
        [34, 167, 132],
        [68, 190, 112],
        [121, 209, 81],
        [189, 222, 38],
        [253, 231, 36],
    ], dtype=np.float32)
    xs = np.linspace(0, 1, len(anchors))
    out = np.zeros((256, 3), dtype=np.uint8)
    for i, t in enumerate(np.linspace(0, 1, 256)):
        rgb = np.array([np.interp(t, xs, anchors[:, c]) for c in range(3)])
        out[i] = np.clip(rgb, 0, 255).astype(np.uint8)
    _VIRIDIS = out
    return out


_detector: Detector | None = None


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector
