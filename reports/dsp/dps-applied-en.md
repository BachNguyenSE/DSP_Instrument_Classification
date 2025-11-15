# DSP-Based Musical Instrument Classification

## Digital Signal Processing Concepts and Theory

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Digital Signal Processing Fundamentals](#digital-signal-processing-fundamentals)
3. [Audio Preprocessing](#audio-preprocessing)
4. [Digital Filtering](#digital-filtering)
5. [Frequency Domain Analysis](#frequency-domain-analysis)
6. [Feature Extraction](#feature-extraction)
7. [Machine Learning Classification](#machine-learning-classification)
8. [System Pipeline](#system-pipeline)

---

## Project Overview

This project develops a **Digital Signal Processing (DSP)-based system** for classifying musical instruments (Drum, Flute, Guitar, Piano, Violin, Tambourine) from audio recordings. The system analyzes unique acoustic characteristics of each instrument through signal processing techniques and machine learning.

**Key Components:**

-    Audio preprocessing and filtering
-    Frequency domain transformations
-    Feature extraction (MFCC, spectral features)
-    Support Vector Machine (SVM) classification

---

## Digital Signal Processing Fundamentals

### What is DSP?

**Digital Signal Processing (DSP)** is the mathematical manipulation of discrete-time signals to extract information, filter noise, or transform signals for analysis. Unlike analog processing, DSP operates on sampled data points.

### Key Concepts

#### 1. **Sampling**

-    **Definition**: Converting a continuous-time analog signal into discrete-time samples
-    **Sampling Rate (Fs)**: Number of samples per second (Hz)
     -    This project uses: **16,000 Hz (16 kHz)**
     -    Higher sampling rates capture more frequency information but require more storage

#### 2. **Nyquist-Shannon Sampling Theorem**

-    **Critical Rule**: To accurately represent a signal, the sampling rate must be at least **twice the highest frequency** present in the signal
-    **Nyquist Frequency**: `Fs/2` = maximum representable frequency
     -    For 16 kHz: Nyquist frequency = **8 kHz**
-    **Aliasing**: If sampling rate is too low, high frequencies appear as lower frequencies (distortion)

#### 3. **Quantization**

-    Converting continuous amplitude values to discrete levels
-    Affects signal quality and dynamic range

---

## Audio Preprocessing

### 1. **Resampling**

**Theory:**

-    Audio files may have different sample rates (22.05 kHz, 44.1 kHz, 48 kHz, etc.)
-    **Standardization** to a fixed rate (16 kHz) ensures consistent processing
-    Uses interpolation/decimation algorithms to change sample rate while preserving frequency content

**Implementation:**

```python
y = librosa.resample(y, orig_sr=orig_sr, target_sr=16000)
```

**Why 16 kHz?**

-    Most musical instrument characteristics are below 8 kHz
-    Reduces computational cost
-    Maintains sufficient frequency resolution for classification

### 2. **Mono Conversion**

**Theory:**

-    Stereo audio has left and right channels
-    For instrument classification, we typically use **mono** (single channel)
-    Mono conversion: average of left/right channels or select one channel
-    Reduces data dimensionality while preserving essential information

**Implementation:**

```python
y, sr = librosa.load(path, sr=None, mono=True)
```

### 3. **Duration Normalization**

**Theory:**

-    Audio files vary in length
-    Machine learning requires **fixed-length** feature vectors
-    **Trimming**: Cut longer audio to fixed duration (e.g., 3 seconds)
-    **Zero-padding**: Extend shorter audio by adding zeros

**Implementation:**

```python
target_len = int(3.0 * sample_rate)  # 3 seconds
if len(y) > target_len:
    y = y[:target_len]  # Trim
elif len(y) < target_len:
    y = np.pad(y, (0, target_len - len(y)))  # Pad with zeros
```

---

## Digital Filtering

### Theory of Digital Filters

**Purpose:** Remove unwanted frequency components or enhance desired frequency ranges

### FIR Bandpass Filter

This project uses a **Finite Impulse Response (FIR) bandpass filter** with frequency range **100 Hz - 8,000 Hz**.

#### 1. **FIR Filter Characteristics**

-    **Impulse Response**: Finite duration (stabilizes after finite samples)
-    **Always Stable**: No feedback, no poles outside unit circle
-    **Linear Phase**: Preserves signal shape, no phase distortion
-    **Easier to Design**: Uses windowing methods

#### 2. **Bandpass Filter Design**

**Purpose:**

-    **Low-cut (100 Hz)**: Remove DC offset, low-frequency noise, rumble
-    **High-cut (8,000 Hz)**: Remove high-frequency noise, preserve signal within Nyquist limit

**Design Method: Windowed FIR (Hamming/Rectangular)**

```python
taps = firwin(
    numtaps=101,           # Filter order (number of coefficients)
    cutoff=[100, 8000],    # Frequency range [low, high] Hz
    pass_zero=False,       # Bandpass (not lowpass/highpass)
    fs=16000               # Sampling frequency
)
```

**Filter Order (numtaps):**

-    Higher order = sharper cutoff, better frequency selectivity
-    Trade-off: More computation, longer delay

#### 3. **Filter Implementation**

**Convolution-based filtering:**

```python
filtered = lfilter(taps, 1.0, signal)
```

-    **Convolution**: Sliding window operation
-    Each output sample = weighted sum of input samples
-    Coefficients (`taps`) determine frequency response

#### 4. **Frequency Response**

-    **Passband**: Frequencies that pass through (100-8000 Hz)
-    **Stopband**: Frequencies that are attenuated
-    **Transition Band**: Region between passband and stopband

---

## Frequency Domain Analysis

### From Time Domain to Frequency Domain

**Time Domain**: Signal amplitude vs. time (waveform)  
**Frequency Domain**: Signal amplitude vs. frequency (spectrum)

### 1. **Discrete Fourier Transform (DFT)**

**Theory:**

-    Decomposes a signal into its constituent frequency components
-    Represents signal as sum of sinusoids at different frequencies
-    Reveals which frequencies are present and their magnitudes

**Mathematical Formula:**

```
X(k) = Σ x(n) · e^(-j2πkn/N)
```

where:

-    `x(n)` = time-domain samples
-    `X(k)` = frequency-domain coefficients
-    `N` = number of samples
-    `k` = frequency bin index

### 2. **Fast Fourier Transform (FFT)**

**Implementation:**

-    **FFT** is an efficient algorithm to compute DFT
-    Reduces complexity from O(N²) to O(N log N)
-    Uses divide-and-conquer approach

**Project Implementation:**

```python
Y = np.fft.fft(signal)              # Compute FFT
freq = np.fft.fftfreq(N, d=1/sr)    # Frequency bins
magnitude = np.abs(Y[:N//2])        # Take positive half (Nyquist)
```

**Key Points:**

-    FFT returns complex numbers: `magnitude` and `phase`
-    Only need positive frequencies (negative are symmetric)
-    Frequency resolution: `Δf = Fs / N`

### 3. **Short-Time Fourier Transform (STFT)**

**Problem with FFT:**

-    FFT assumes signal is stationary (properties don't change over time)
-    Music signals are **non-stationary** (frequency content changes)

**Solution: STFT**

-    Divide signal into short overlapping windows
-    Compute FFT on each window
-    Result: **Time-frequency representation** (spectrogram)

**Parameters:**

-    **n_fft = 1024**: Window size for FFT (frequency resolution)
-    **hop_length = 512**: Step size between windows (time resolution)

**Trade-off:**

-    Larger windows = better frequency resolution, worse time resolution
-    Smaller windows = better time resolution, worse frequency resolution

**Implementation:**

```python
stft = librosa.stft(signal, n_fft=1024, hop_length=512)
S_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
```

**Spectrogram:**

-    2D representation: Time (x-axis) vs. Frequency (y-axis) vs. Magnitude (color)
-    Visualizes how frequency content evolves over time

---

## Feature Extraction

Feature extraction converts raw audio signals into numerical vectors that capture distinctive characteristics of each instrument.

### 1. **MFCC (Mel-Frequency Cepstral Coefficients)**

**Why MFCC?**

-    Captures timbral characteristics (tone quality, texture)
-    Based on human auditory perception
-    Compact representation (typically 13-20 coefficients)
-    Effective for instrument classification

#### MFCC Computation Pipeline:

**Step 1: Pre-emphasis**

-    High-frequency boost to emphasize important spectral details
-    `y[n] = x[n] - α·x[n-1]` (typically α = 0.97)

**Step 2: Windowing**

-    Divide signal into frames (typically 25 ms, 50% overlap)
-    Apply window function (Hamming/Hann) to reduce spectral leakage

**Step 3: FFT**

-    Compute power spectrum for each frame

**Step 4: Mel-scale Filterbank**

-    **Mel scale**: Perceptual scale of pitch (humans perceive pitch logarithmically)
-    Apply triangular filterbank on Mel scale (typically 26-40 filters)
-    Convert frequency (Hz) to Mel: `mel = 2595·log₁₀(1 + f/700)`

**Step 5: Logarithm**

-    Take log of filterbank energies
-    Mimics human perception of loudness (logarithmic)

**Step 6: DCT (Discrete Cosine Transform)**

-    Compress filterbank outputs into fewer coefficients
-    First 13-20 coefficients (MFCCs) capture most information
-    Higher coefficients represent fine spectral details (often noise)

**Project Implementation:**

```python
mfcc = librosa.feature.mfcc(y=signal, sr=16000, n_mfcc=20)
mfcc_mean = mfcc.mean(axis=1)  # Average across time
mfcc_std = mfcc.std(axis=1)    # Standard deviation (captures variation)
```

**Feature Vector:**

-    20 mean MFCCs + 20 std MFCCs = **40 MFCC features**

### 2. **Spectral Features**

#### A. **Spectral Centroid**

**Definition:** The "brightness" of a sound - weighted average frequency  
**Formula:** `centroid = Σ(f(k) · magnitude(k)) / Σ(magnitude(k))`

**Interpretation:**

-    **High centroid** (e.g., flute, violin): Bright, sharp sound
-    **Low centroid** (e.g., piano, bass): Dark, warm sound

**Instrument Characteristics:**

-    Violin: ~2000-4000 Hz
-    Flute: ~2000-5000 Hz
-    Guitar: ~1000-3000 Hz
-    Piano: ~500-2000 Hz
-    Drum/Tambourine: Broad frequency range (percussive)

#### B. **Spectral Bandwidth**

**Definition:** Spread of spectrum around centroid (measure of spectral concentration)  
**Formula:** `bandwidth = √(Σ((f(k) - centroid)² · magnitude(k)) / Σ(magnitude(k)))`

**Interpretation:**

-    **Narrow bandwidth**: Concentrated frequency content (e.g., pure tones)
-    **Wide bandwidth**: Spread frequency content (e.g., noise-like sounds, percussive)

#### C. **Spectral Rolloff**

**Definition:** Frequency below which a certain percentage (typically 85%) of spectral energy is contained

**Interpretation:**

-    Measures spectral shape and energy distribution
-    Lower rolloff: Energy concentrated at low frequencies
-    Higher rolloff: Energy spread across higher frequencies

#### D. **Zero Crossing Rate (ZCR)**

**Definition:** Rate at which signal crosses zero amplitude per unit time  
**Formula:** Count zero crossings / total samples

**Interpretation:**

-    **High ZCR**: Noisy, high-frequency content (e.g., cymbals, consonants)
-    **Low ZCR**: Smooth, low-frequency content (e.g., vowels, sustained tones)
-    Useful for distinguishing percussive vs. sustained instruments

**Project Implementation:**

```python
centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y=signal)

# Aggregate: mean and std across time frames
features = {
    "centroid_mean": centroid.mean(),
    "centroid_std": centroid.std(),
    "bandwidth_mean": bandwidth.mean(),
    "bandwidth_std": bandwidth.std(),
    "rolloff_mean": rolloff.mean(),
    "rolloff_std": rolloff.std(),
    "zcr_mean": zcr.mean(),
    "zcr_std": zcr.std()
}
```

**Total Spectral Features: 8** (4 features × 2 statistics)

### 3. **Complete Feature Vector**

**Final Feature Vector Size:**

-    40 MFCC features (20 mean + 20 std)
-    8 Spectral features (4 features × 2 statistics)
-    **Total: 48 features per audio sample**

---

## Machine Learning Classification

### Feature Normalization

**StandardScaler (Z-score normalization):**

```python
z = (x - μ) / σ
```

where:

-    `μ` = mean of feature
-    `σ` = standard deviation of feature

**Why Normalize?**

-    Features have different scales (MFCC vs. spectral centroid)
-    Prevents features with larger magnitudes from dominating
-    Improves convergence and accuracy of SVM

### Support Vector Machine (SVM)

#### Theory

**Concept:** Find optimal hyperplane that separates classes with maximum margin

**Key Components:**

1. **Support Vectors**: Data points closest to decision boundary
2. **Margin**: Distance between hyperplane and nearest points
3. **Kernel Trick**: Map data to higher-dimensional space for non-linear separation

#### RBF (Radial Basis Function) Kernel

**Formula:** `K(x₁, x₂) = exp(-γ·||x₁ - x₂||²)`

**Parameters:**

-    **C = 10.0**: Penalty for misclassification
     -    Higher C: Stricter, may overfit
     -    Lower C: More flexible, may underfit
-    **gamma = "scale"**: Kernel coefficient
     -    Controls influence of individual samples
     -    "scale" = automatic tuning based on feature variance

**Why RBF Kernel?**

-    Handles non-linear decision boundaries
-    Effective for high-dimensional feature spaces
-    Good performance on audio classification tasks

**Project Configuration:**

```python
clf = SVC(kernel="rbf", C=10.0, gamma="scale")
```

---

## System Pipeline

### Training Pipeline

```
Audio Files (WAV)
    ↓
1. Load & Resample (→ 16 kHz, mono)
    ↓
2. Duration Normalization (trim/pad → 3s)
    ↓
3. FIR Bandpass Filter (100-8000 Hz)
    ↓
4. Feature Extraction
    ├─ MFCC (20 coefficients × 2 stats = 40)
    └─ Spectral Features (4 features × 2 stats = 8)
    ↓
5. Feature Vector (48 dimensions)
    ↓
6. StandardScaler (normalization)
    ↓
7. SVM Training (RBF kernel)
    ↓
Saved Model + Scaler
```

### Inference Pipeline

```
Test Audio File
    ↓
1. Load & Preprocess (same as training)
    ↓
2. Filter & Extract Features
    ↓
3. Normalize (using saved scaler)
    ↓
4. Predict (using saved SVM model)
    ↓
Instrument Classification Result
```

### Why This Pipeline Works

1. **Preprocessing**: Standardizes input → consistent analysis
2. **Filtering**: Removes noise → cleaner features
3. **FFT/STFT**: Reveals frequency content → instrument timbre
4. **MFCC**: Captures timbral characteristics → perceptually relevant
5. **Spectral Features**: Captures brightness, bandwidth → distinguishing features
6. **SVM**: Learns decision boundaries → robust classification

---

## Summary of DSP Concepts

### Core DSP Techniques

1. **Sampling & Quantization**: Analog → Digital conversion
2. **Nyquist Theorem**: Sampling rate requirements
3. **Digital Filtering**: FIR bandpass filtering
4. **Frequency Domain Analysis**: FFT, STFT, Spectrograms
5. **Mel-Scale Perception**: MFCC computation
6. **Feature Engineering**: Statistical aggregation
7. **Normalization**: Feature scaling
8. **Machine Learning**: SVM classification

### Mathematical Foundations

-    **Linear Algebra**: Vector operations, matrix transformations
-    **Signal Processing**: Convolution, Fourier transforms, filtering
-    **Statistics**: Mean, standard deviation, normalization
-    **Optimization**: SVM margin maximization

### Practical Applications

-    Music information retrieval
-    Audio classification systems
-    Speech recognition
-    Audio quality enhancement
-    Acoustic analysis

---

## Conclusion

This project demonstrates how **Digital Signal Processing** techniques can extract meaningful features from audio signals for instrument classification. By combining:

-    **Preprocessing** (resampling, filtering)
-    **Frequency analysis** (FFT, STFT)
-    **Perceptually-motivated features** (MFCC)
-    **Statistical features** (spectral properties)
-    **Machine learning** (SVM)

We create a robust system capable of distinguishing between different musical instruments based on their unique acoustic signatures.

**Key Insight**: Each instrument has a distinct "fingerprint" in the frequency domain, which can be captured and classified through appropriate DSP techniques.

---

## References

-    Oppenheim & Schafer: _Discrete-Time Signal Processing_
-    Rabiner & Schafer: _Digital Processing of Speech Signals_
-    Librosa Documentation: Audio analysis library
-    Scikit-learn: Machine learning library

---

_End of Presentation_
