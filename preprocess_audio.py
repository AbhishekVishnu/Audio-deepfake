import os
import librosa
import numpy as np
import soundfile as sf

# ---------------- CONFIG ---------------- #
SAMPLE_RATE = 16000
DURATION = 5  # seconds
N_MELS = 128

REAL_PATH = "dataset/real"
FAKE_PATH = "dataset/fake"

OUTPUT_WAVE = "output/waveforms"
OUTPUT_SPEC = "output/spectrograms"
# ---------------------------------------- #


def load_audio(file_path, sr=SAMPLE_RATE, duration=DURATION):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    max_len = sr * duration

    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = np.pad(audio, (0, max_len - len(audio)))

    return audio


def remove_silence(audio):
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=20)
    return trimmed_audio


def normalize_audio(audio):
    if np.max(np.abs(audio)) == 0:
        return audio
    return audio / np.max(np.abs(audio))


def get_spectrogram(audio, sr=SAMPLE_RATE):
    spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db


def preprocess_audio(file_path):
    audio = load_audio(file_path)
    audio = remove_silence(audio)
    audio = normalize_audio(audio)

    waveform = audio
    spectrogram = get_spectrogram(audio)

    return waveform, spectrogram


def save_features(waveform, spectrogram, label, file_name):
    wave_dir = os.path.join(OUTPUT_WAVE, label)
    spec_dir = os.path.join(OUTPUT_SPEC, label)

    os.makedirs(wave_dir, exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)

    np.save(os.path.join(wave_dir, file_name + ".npy"), waveform)
    np.save(os.path.join(spec_dir, file_name + ".npy"), spectrogram)


def process_dataset(folder_path, label):
    for file in os.listdir(folder_path):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(folder_path, file)
        file_name = os.path.splitext(file)[0]

        try:
            waveform, spectrogram = preprocess_audio(file_path)
            save_features(waveform, spectrogram, label, file_name)
            print(f"[OK] Processed: {file}")
        except Exception as e:
            print(f"[ERROR] {file} → {e}")


def main():
    print("Starting Audio Preprocessing...")

    process_dataset(REAL_PATH, "real")
    process_dataset(FAKE_PATH, "fake")

    print("Preprocessing Completed Successfully!")


if __name__ == "__main__":
    main()