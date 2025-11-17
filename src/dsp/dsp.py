# dsp.py (moved to dsp/dsp.py)
from typing import Tuple

import numpy as np
import librosa
from scipy.signal import firwin, lfilter

from config.config import SAMPLE_RATE

def compute_fft(signal: np.ndarray, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính FFT và trả về (freq, magnitude) chỉ lấy nửa phổ dương.
    """
    N = len(signal)
    Y = np.fft.fft(signal)
    freq = np.fft.fftfreq(N, d=1.0 / sr)

    half = N // 2
    return freq[:half], np.abs(Y[:half])

def compute_spectrogram(
    signal: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Tính STFT và trả về spectrogram (dB).
    Dùng sau này nếu muốn vẽ hình hoặc cho CNN.
    """
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return S_db

def design_bandpass_fir(
    sr: int = SAMPLE_RATE,
    lowcut: float = 100.0,
    highcut: float = 8000.0,
    numtaps: int = 101,
) -> np.ndarray:
    """
    Thiết kế FIR bandpass filter [lowcut, highcut] Hz.
    """
    # Lưu ý: highcut phải < sr/2 (theo Nyquist)
    if highcut >= sr / 2:
        highcut = sr / 2 - 1

    taps = firwin(
        numtaps=numtaps,
        cutoff=[lowcut, highcut],
        pass_zero=False,
        fs=sr,
    )
    return taps.astype(np.float32)

def apply_filter(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """
    Áp dụng FIR filter lên tín hiệu.
    """
    filtered = lfilter(taps, 1.0, signal)
    return filtered.astype(np.float32)
