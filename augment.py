"""Audio augmentation utilities for the robustness demo.

Each augment loads a wav, applies a perturbation, writes a new wav, and returns the path.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

SR = 16000

AUGMENT_KINDS = ("noise", "lowband", "pitch", "reverb", "speed")


def _load(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    return audio.astype(np.float32)


def _save(audio: np.ndarray, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"aug_{uuid.uuid4().hex}.wav"
    sf.write(path, audio, SR)
    return path


def add_noise(audio: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    sig_pow = float(np.mean(audio ** 2)) + 1e-12
    noise = np.random.randn(len(audio)).astype(np.float32)
    noise_pow = float(np.mean(noise ** 2)) + 1e-12
    target_noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise *= np.sqrt(target_noise_pow / noise_pow)
    return audio + noise


def lowband_telephone(audio: np.ndarray) -> np.ndarray:
    """Simulate telephone-band: down-sample to 8 kHz, back up, with bandpass-ish flavour."""
    down = librosa.resample(audio, orig_sr=SR, target_sr=8000)
    up = librosa.resample(down, orig_sr=8000, target_sr=SR)
    # Mild high-pass to mimic telephone roll-off
    return up - np.mean(up)


def pitch_shift(audio: np.ndarray, semitones: float = 2.0) -> np.ndarray:
    return librosa.effects.pitch_shift(y=audio, sr=SR, n_steps=semitones)


def reverb(audio: np.ndarray, decay: float = 0.4) -> np.ndarray:
    """Cheap synthetic reverb: convolve with an exponentially decaying impulse train."""
    length = int(0.4 * SR)
    impulse = np.zeros(length, dtype=np.float32)
    impulse[0] = 1.0
    for i, t in enumerate(np.linspace(0.02, 0.4, 8)):
        idx = int(t * SR)
        if idx < length:
            impulse[idx] = decay ** (i + 1)
    out = fftconvolve(audio, impulse)[: len(audio)]
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak).astype(np.float32) * float(np.max(np.abs(audio)) or 1.0)


def speed_change(audio: np.ndarray, rate: float = 1.1) -> np.ndarray:
    return librosa.effects.time_stretch(y=audio, rate=rate)


def apply(kind: str, src_path: str, dest_dir: Path) -> Path:
    audio = _load(src_path)
    if kind == "noise":
        out = add_noise(audio, snr_db=10.0)
    elif kind == "lowband":
        out = lowband_telephone(audio)
    elif kind == "pitch":
        out = pitch_shift(audio, semitones=2.0)
    elif kind == "reverb":
        out = reverb(audio)
    elif kind == "speed":
        out = speed_change(audio, rate=1.1)
    else:
        raise ValueError(f"unknown augment kind: {kind}")
    return _save(out, dest_dir)
