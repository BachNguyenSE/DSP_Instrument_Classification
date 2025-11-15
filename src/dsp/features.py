# features.py (moved to dsp/features.py)
from typing import Dict

import numpy as np
import librosa

from config.config import SAMPLE_RATE, N_MFCC

def extract_mfcc(
    signal: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
) -> np.ndarray:
    """
    Tính MFCC (n_mfcc x T)
    """
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,  
    )
    return mfcc

def extract_spectral_features(
    signal: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> Dict[str, float]:
    """
    Tính một số đặc trưng phổ cơ bản: centroid, bandwidth, rolloff, zcr.
    Trả về dict {tên_feature: giá_trị_scalar}
    """
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=signal)

    feats = {
        "centroid_mean": float(centroid.mean()),
        "centroid_std": float(centroid.std()),
        "bandwidth_mean": float(bandwidth.mean()),
        "bandwidth_std": float(bandwidth.std()),
        "rolloff_mean": float(rolloff.mean()),
        "rolloff_std": float(rolloff.std()),
        "zcr_mean": float(zcr.mean()),
        "zcr_std": float(zcr.std()),
    }
    return feats

def build_feature_vector(
    signal: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Gộp MFCC (mean + std) và spectral features thành 1 vector 1D.
    Đây là input chính cho model ML.
    """
    # MFCC
    mfcc = extract_mfcc(signal, sr=sr)
    # mfcc shape: (n_mfcc, T)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # Spectral features
    spec_dict = extract_spectral_features(signal, sr=sr)
    spec_values = np.array(list(spec_dict.values()), dtype=np.float32)

    # Gộp lại
    feat_vec = np.concatenate(
        [mfcc_mean.astype(np.float32), mfcc_std.astype(np.float32), spec_values],
        axis=0,
    )
    return feat_vec
