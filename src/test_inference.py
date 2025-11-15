# src/test_inference.py

from pathlib import Path
import sys

import numpy as np
import librosa
import joblib

from .config import (
    SAMPLE_RATE,
    AUDIO_DURATION,
    INSTRUMENT_CLASSES,
    PROJECT_ROOT,
    TEST_DIR,
)
from .dsp import design_bandpass_fir, apply_filter
from .features import build_feature_vector


# Đường dẫn model & scaler đã save ở train_ml.py
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "svm_instrument.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"


def load_audio_any(path: Path):
    """
    Load file .wav / .mp3 bất kỳ, resample về SAMPLE_RATE
    và cắt/pad về AUDIO_DURATION giây.
    """
    y, sr = librosa.load(str(path), sr=None, mono=True)

    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    target_len = int(AUDIO_DURATION * sr)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    return y.astype(np.float32), sr


def predict_file(audio_path: Path):
    # Load model + scaler
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Thiết kế filter
    taps = design_bandpass_fir(sr=SAMPLE_RATE)

    # Load audio bất kỳ
    signal, sr = load_audio_any(audio_path)

    # Filter
    filtered = apply_filter(signal, taps)

    # Feature vector
    feat_vec = build_feature_vector(filtered, sr=sr).reshape(1, -1)

    # Scale
    feat_scaled = scaler.transform(feat_vec)

    # Predict
    pred_idx = int(clf.predict(feat_scaled)[0])
    pred_label = INSTRUMENT_CLASSES[pred_idx]

    print(f"{audio_path.name} → dự đoán nhạc cụ: {pred_label}")


def main():
    # Cho phép: python3 -m src.test_inference path/to/file.wav
    if len(sys.argv) == 2:
        audio_path = Path(sys.argv[1])
        if not audio_path.exists():
            print(f"Không tìm thấy file: {audio_path}")
            sys.exit(1)
        predict_file(audio_path)
        return

    # Nếu không truyền arg → quét thư mục TEST_DIR
    print(f"[INFO] Đang tìm file test trong: {TEST_DIR}")
    if not TEST_DIR.exists():
        print(f"Thư mục test không tồn tại: {TEST_DIR}")
        sys.exit(1)

    test_files = [
        p for p in TEST_DIR.iterdir()
        if p.suffix.lower() in (".wav", ".mp3")
    ]

    if not test_files:
        print("Thư mục test trống hoặc không có file .wav/.mp3. "
              "Hãy bỏ vài file vào data/test rồi chạy lại.")
        sys.exit(0)

    for f in sorted(test_files):
        predict_file(f)


if __name__ == "__main__":
    main()
