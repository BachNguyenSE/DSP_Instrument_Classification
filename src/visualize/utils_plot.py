# utils_plot.py (moved to visualize/utils_plot.py)
import matplotlib.pyplot as plt
import librosa.display

def plot_waveform(signal, sr, title="Waveform"):
    plt.figure()
    librosa.display.waveshow(signal, sr=sr)
    plt.title(title)

def plot_spectrogram(S_db, sr, title="Spectrogram"):
    plt.figure()
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
