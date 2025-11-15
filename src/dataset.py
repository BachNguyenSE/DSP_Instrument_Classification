# src/dataset.py

import os
from typing import List, Tuple

import librosa
import numpy as np

from .config import SAMPLE_RATE, AUDIO_DURATION, RAW_DATA_DIR, INSTRUMENT_CLASSES

# Map label string ↔ id int
label2id = {name: idx for idx, name in enumerate(INSTRUMENT_CLASSES)}
id2label = {v: k for k, v in label2id.items()}


def list_audio_files(root_dir=RAW_DATA_DIR) -> List[Tuple[str, str]]:
    """
    Duyệt qua data/raw/<instrument>/*.wav
    Trả về list (path, label_str)
    """
    files: List[Tuple[str, str]] = []

    for label in INSTRUMENT_CLASSES:
        class_dir = root_dir / label
        if not class_dir.exists():
            print(f"[WARN] Class dir not found: {class_dir}")
            continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith(".wav"):
                full_path = class_dir / fname
                files.append((str(full_path), label))

    print(f"[INFO] Found {len(files)} audio files.")
    return files


def load_audio(
    path: str,
    duration: float = AUDIO_DURATION,
    sr: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """
    Load audio, resample về sr, cắt/pad về duration (giây).
    """
    # mono=True để lấy 1 kênh
    y, orig_sr = librosa.load(path, sr=None, mono=True)

    # Resample nếu khác sample_rate
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

    target_len = int(duration * sr)

    if len(y) > target_len:
        # Cắt bớt
        y = y[:target_len]
    elif len(y) < target_len:
        # Pad thêm 0
        pad_len = target_len - len(y)
        y = np.pad(y, (0, pad_len))

    return y.astype(np.float32), sr
