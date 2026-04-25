"""Fine-tune the offline Keras fusion model on dataset/ + feedback/.

The model has two inputs (mel-spectrogram + raw waveform) derived from a
3-second 16 kHz mono clip — same preprocessing as the original. We just
load the existing `.keras` checkpoint and continue fitting; no need to
rebuild the architecture.

Usage:
  python train.py                           # fine-tune for 5 epochs, save versioned
  python train.py --epochs 10 --batch 32    # more epochs, bigger batches
  python train.py --feedback-only           # train ONLY on feedback/ (curriculum)
  python train.py --no-feedback             # ignore feedback/ — train on dataset/
  python train.py --val-split 0.15
  python train.py --replace                 # also overwrite fusion_model_final.keras

The new checkpoint is written to model/fusion_model_final_v{N}.keras and the
ensemble picks it up on next server restart if you set:
  DEEPFAKE_KERAS_MODEL=model/fusion_model_final_v3.keras
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("train")

BASE = Path(__file__).resolve().parent
DATASET_DIR = BASE / "dataset"
FEEDBACK_DIR = BASE / "feedback"
MODEL_DIR = BASE / "model"
CURRENT = MODEL_DIR / "fusion_model_final.keras"

# Match the inference-time preprocessing exactly.
SR = 16000
DURATION_S = 3
SAMPLES = SR * DURATION_S
N_MELS = 128
N_FFT = 2048
HOP = 512
SPEC_FRAMES = 94  # 48000 / 512 + 1 ≈ 94


@dataclass
class Sample:
    path: Path
    label: int          # 1 = fake, 0 = real


def collect_samples(use_dataset: bool, use_feedback: bool) -> list[Sample]:
    out: list[Sample] = []
    if use_dataset:
        real_dir = DATASET_DIR / "real"
        fake_dir = DATASET_DIR / "fake"
        if real_dir.exists():
            out += [Sample(p, 0) for p in real_dir.iterdir() if p.suffix.lower() == ".wav"]
        if fake_dir.exists():
            out += [Sample(p, 1) for p in fake_dir.iterdir() if p.suffix.lower() == ".wav"]
    if use_feedback:
        for p in (FEEDBACK_DIR / "real").glob("*"):
            if p.is_file():
                out.append(Sample(p, 0))
        for p in (FEEDBACK_DIR / "fake").glob("*"):
            if p.is_file():
                out.append(Sample(p, 1))
    return out


def load_clip(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (spec, wave) or None if the file is unreadable."""
    import librosa
    try:
        audio, _ = librosa.load(str(path), sr=SR, mono=True)
    except Exception as exc:
        log.warning("skip %s: %s", path.name, exc)
        return None
    if audio.size == 0:
        return None
    # Centre crop / pad to 3 s
    if len(audio) >= SAMPLES:
        start = (len(audio) - SAMPLES) // 2
        audio = audio[start:start + SAMPLES]
    else:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    audio = audio.astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    if mel_db.shape[1] >= SPEC_FRAMES:
        mel_db = mel_db[:, :SPEC_FRAMES]
    else:
        mel_db = np.pad(mel_db, ((0, 0), (0, SPEC_FRAMES - mel_db.shape[1])))
    return mel_db, audio


class FusionSequence:
    """tf.keras.utils.Sequence that batches (spec, wave) → label."""

    def __init__(self, samples: list[Sample], batch: int, shuffle: bool = True) -> None:
        from tensorflow.keras.utils import Sequence  # local import keeps the file importable
        self._Sequence = Sequence
        self.samples = list(samples)
        self.batch = batch
        self.shuffle = shuffle
        self._build()

    def _build(self):
        if self.shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return (len(self.samples) + self.batch - 1) // self.batch

    def __getitem__(self, idx):
        chunk = self.samples[idx * self.batch:(idx + 1) * self.batch]
        specs, waves, labels = [], [], []
        for s in chunk:
            r = load_clip(s.path)
            if r is None:
                continue
            spec, wave = r
            specs.append(spec)
            waves.append(wave)
            labels.append(s.label)
        if not specs:
            # Defensive: keep batch shape consistent for keras
            specs = [np.zeros((N_MELS, SPEC_FRAMES), dtype=np.float32)]
            waves = [np.zeros(SAMPLES, dtype=np.float32)]
            labels = [0]
        spec_b = np.stack(specs)[..., np.newaxis]    # (B, 128, 94, 1)
        wave_b = np.stack(waves)[..., np.newaxis]    # (B, 48000, 1)
        y = np.array(labels, dtype=np.float32)
        return [spec_b, wave_b], y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.samples)


def make_keras_seq(samples: list[Sample], batch: int, shuffle: bool):
    """Return a real keras.Sequence (FusionSequence wraps it for typing/logic)."""
    # Build a true Sequence subclass by combining at runtime so tf import is lazy.
    import tensorflow as tf

    class _Seq(tf.keras.utils.Sequence):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i): return self.base[i]
        def on_epoch_end(self): self.base.on_epoch_end()

    return _Seq(FusionSequence(samples, batch, shuffle))


def next_version(stem: str = "fusion_model_final") -> Path:
    existing = sorted(MODEL_DIR.glob(f"{stem}_v*.keras"))
    if not existing:
        return MODEL_DIR / f"{stem}_v1.keras"
    nums = []
    for p in existing:
        try:
            nums.append(int(p.stem.split("_v")[-1]))
        except ValueError:
            pass
    return MODEL_DIR / f"{stem}_v{max(nums) + 1}.keras"


def class_weights(samples: list[Sample]) -> dict:
    n_real = sum(1 for s in samples if s.label == 0) or 1
    n_fake = sum(1 for s in samples if s.label == 1) or 1
    total = n_real + n_fake
    return {0: total / (2 * n_real), 1: total / (2 * n_fake)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4, help="learning rate (lower → fine-tune)")
    ap.add_argument("--feedback-only", action="store_true")
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--from-scratch", action="store_true",
                    help="build a new fusion model instead of loading the existing one")
    ap.add_argument("--replace", action="store_true",
                    help="also overwrite model/fusion_model_final.keras with the new weights")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total samples for quick smoke tests")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    use_ds = not args.feedback_only
    use_fb = not args.no_feedback
    samples = collect_samples(use_ds, use_fb)
    if args.limit:
        random.shuffle(samples)
        samples = samples[:args.limit]
    if not samples:
        sys.exit("No training samples found. Populate dataset/ or feedback/.")

    n_real = sum(1 for s in samples if s.label == 0)
    n_fake = sum(1 for s in samples if s.label == 1)
    log.info("samples: %d real / %d fake (total %d)", n_real, n_fake, len(samples))
    if n_real == 0 or n_fake == 0:
        sys.exit("Need at least one example of each class.")

    # Train/val split (stratified by class)
    real = [s for s in samples if s.label == 0]
    fake = [s for s in samples if s.label == 1]
    random.shuffle(real); random.shuffle(fake)
    cut_real = max(1, int(len(real) * args.val_split))
    cut_fake = max(1, int(len(fake) * args.val_split))
    val = real[:cut_real] + fake[:cut_fake]
    train = real[cut_real:] + fake[cut_fake:]
    random.shuffle(train); random.shuffle(val)
    log.info("split: train=%d  val=%d", len(train), len(val))

    # Lazy TF import (after env vars are set)
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam

    if args.from_scratch or not CURRENT.exists() or CURRENT.stat().st_size < 1024:
        log.info("Building fresh fusion model architecture")
        model = build_fusion_model()
    else:
        log.info("Resuming from %s", CURRENT)
        model = load_model(str(CURRENT), compile=False)

    model.compile(optimizer=Adam(learning_rate=args.lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

    log.info("model summary:")
    model.summary(print_fn=log.info)

    train_seq = make_keras_seq(train, args.batch, shuffle=True)
    val_seq = make_keras_seq(val, args.batch, shuffle=False)

    cw = class_weights(train)
    log.info("class weights: %s", cw)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_auc", mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_loss"),
    ]

    t0 = time.time()
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        class_weight=cw,
        callbacks=callbacks,
        verbose=2,
    )
    log.info("training done in %.1fs", time.time() - t0)

    # Save versioned + optionally replace
    out = next_version()
    model.save(str(out))
    log.info("saved %s", out)
    if args.replace:
        model.save(str(CURRENT))
        log.info("also wrote %s", CURRENT)

    # Print final val metrics for convenience
    val_metrics = model.evaluate(val_seq, verbose=0)
    names = ["loss", "accuracy", "auc"]
    print()
    print("┌── Final validation " + "─" * 36)
    for n, v in zip(names, val_metrics):
        print(f"│ {n:<10} {v:.4f}")
    print(f"│ samples   train={len(train)}  val={len(val)}")
    print(f"│ saved     {out.name}")
    print(f"└" + "─" * 56)

    return 0


def build_fusion_model():
    """A reasonable fusion architecture if no checkpoint exists.

    Two-branch CNN: spec branch (Conv2D) + waveform branch (Conv1D) → concat → dense.
    Inputs match the original (128, 94, 1) and (48000, 1).
    """
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Input, Conv2D, Conv1D, BatchNormalization, MaxPool2D, MaxPool1D,
        Flatten, Dense, Dropout, Concatenate,
    )
    from tensorflow.keras.models import Model

    spec_in = Input(shape=(N_MELS, SPEC_FRAMES, 1), name="spec_in")
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(spec_in)
    x = BatchNormalization()(x); x = MaxPool2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x); x = MaxPool2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x); x = MaxPool2D((2, 2))(x)
    x = Flatten()(x)

    wave_in = Input(shape=(SAMPLES, 1), name="wave_in")
    y = Conv1D(32, 80, strides=4, padding="same", activation="relu")(wave_in)
    y = BatchNormalization()(y); y = MaxPool1D(4)(y)
    y = Conv1D(64, 7, padding="same", activation="relu")(y)
    y = BatchNormalization()(y); y = MaxPool1D(4)(y)
    y = Conv1D(128, 5, padding="same", activation="relu")(y)
    y = BatchNormalization()(y); y = MaxPool1D(4)(y)
    y = Flatten()(y)

    z = Concatenate()([x, y])
    z = Dense(128, activation="relu")(z)
    z = Dropout(0.3)(z)
    z = Dense(64, activation="relu")(z)
    out = Dense(1, activation="sigmoid")(z)

    return Model(inputs=[spec_in, wave_in], outputs=out, name="fusion_model")


if __name__ == "__main__":
    sys.exit(main())
