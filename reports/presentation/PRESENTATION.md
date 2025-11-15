# DSP-Based Musical Instrument Classification

## Presentation Slides

---

## Slide 1: Title Slide

# DSP-Based Musical Instrument Classification

**Digital Signal Processing Project**

_Using Traditional DSP Techniques and Machine Learning_

---

## Slide 2: Problem Statement & Objectives

### Problem Statement

-    Automatic classification of musical instruments from audio recordings
-    Challenge: Extract meaningful acoustic features to distinguish instruments
-    Need for interpretable, efficient solution

### Objectives

1. Develop complete DSP-based pipeline
2. Implement SVM classification system
3. Create interactive visualization interface
4. Achieve robust performance (85-90% accuracy)

---

## Slide 3: System Overview / Architecture

### System Architecture

```
User Interface (Streamlit)
         ↓
Audio Preprocessing
  (Resample, Normalize)
         ↓
DSP Processing
  (Filter, FFT, STFT)
         ↓
Feature Extraction
  (MFCC, Spectral)
         ↓
Classification (SVM)
         ↓
Results & Visualization
```

**Key Components:**

-    6 Instrument Classes: Drum, Flute, Guitar, Piano, Violin, Tambourine
-    48-Dimensional Feature Vector
-    Real-time Web Interface

---

## Slide 4: DSP Pipeline Overview

### Complete Processing Pipeline

1. **Audio Loading** → Resample to 16 kHz, convert to mono
2. **Duration Normalization** → Trim/pad to 3 seconds
3. **Bandpass Filtering** → FIR filter (100-8000 Hz)
4. **Frequency Analysis** → FFT and STFT
5. **Feature Extraction** → MFCC + Spectral features
6. **Classification** → SVM with RBF kernel

**Why 16 kHz?**

-    Most instrument characteristics below 8 kHz (Nyquist limit)
-    Reduces computational cost
-    Maintains sufficient resolution

---

## Slide 5: Preprocessing & Filtering

### Audio Preprocessing

**Resampling:**

-    Standardize to 16 kHz sampling rate
-    Nyquist frequency = 8 kHz
-    Prevents aliasing

**Duration Normalization:**

-    Fixed 3-second clips
-    Trimming for longer audio
-    Zero-padding for shorter audio

### FIR Bandpass Filter

**Specifications:**

-    Frequency Range: 100-8000 Hz
-    Filter Order: 101 taps
-    Design: Windowed FIR (Hamming)

**Purpose:**

-    Remove DC offset and low-frequency noise
-    Eliminate high-frequency artifacts
-    Preserve relevant frequency content

---

## Slide 6: Frequency Domain Analysis

### Fast Fourier Transform (FFT)

**Purpose:** Convert time-domain signal to frequency domain

**Key Points:**

-    Reveals frequency components and harmonics
-    Complexity: O(N log N)
-    Only positive frequencies needed (symmetric)

### Short-Time Fourier Transform (STFT)

**Purpose:** Handle non-stationary music signals

**Parameters:**

-    `n_fft = 1024`: Window size (frequency resolution)
-    `hop_length = 512`: Step size (time resolution)

**Result:** Spectrogram - 2D time-frequency representation

**Trade-off:** Frequency resolution vs. Time resolution

---

## Slide 7: Feature Extraction

### MFCC (Mel-Frequency Cepstral Coefficients)

**Why MFCC?**

-    Captures timbral characteristics
-    Based on human auditory perception
-    Compact representation (20 coefficients)

**Computation Pipeline:**

1. Pre-emphasis
2. Windowing
3. FFT
4. Mel-scale filterbank
5. Logarithm
6. DCT

**Features:** 20 mean + 20 std = **40 MFCC features**

### Spectral Features

**Four Types:**

-    **Spectral Centroid**: Brightness of sound
-    **Spectral Bandwidth**: Spread of spectrum
-    **Spectral Rolloff**: Energy distribution
-    **Zero Crossing Rate**: Percussive vs. sustained

**Features:** 4 features × 2 stats = **8 spectral features**

**Total Feature Vector: 48 dimensions**

---

## Slide 8: Machine Learning

### Support Vector Machine (SVM)

**Algorithm:** Find optimal hyperplane with maximum margin

**Kernel:** RBF (Radial Basis Function)

-    Handles non-linear boundaries
-    Effective for high-dimensional features
-    Formula: `K(x₁, x₂) = exp(-γ·||x₁ - x₂||²)`

**Parameters:**

-    **C = 10.0**: Penalty for misclassification
-    **gamma = "scale"**: Automatic kernel coefficient

### Feature Normalization

**StandardScaler (Z-score):**

-    Normalizes features to same scale
-    Prevents feature dominance
-    Improves SVM convergence

**Training Configuration:**

-    Train/Test Split: 80/20 (stratified)
-    Random Seed: 42 (reproducibility)

---

## Slide 9: Implementation Highlights

### Code Structure

```
src/
├── config/        # Configuration constants
├── data/          # Data loading & preprocessing
├── dsp/           # DSP operations & features
├── train/         # Training pipeline
├── inference/     # Prediction/testing
└── app_streamlit.py  # Web interface
```

### Key Modules

**DSP Module (`dsp.py`):**

-    FFT computation
-    Spectrogram generation
-    FIR filter design & application

**Features Module (`features.py`):**

-    MFCC extraction
-    Spectral feature computation
-    Feature vector construction

**Training Module (`train_ml.py`):**

-    Complete training pipeline
-    Model evaluation
-    Model serialization

---

## Slide 10: Results & Performance Metrics

### Model Performance

**Overall Accuracy:**

-    **Training:** ~90-95%
-    **Validation/Test:** ~85-90%
-    **Inference Time:** <1 second per file

### Classification Results

**Well-Performing Classes:**

-    Piano, Violin, Guitar show high precision/recall

**Challenging Pairs:**

-    Drum vs. Tambourine (both percussive)
-    Flute vs. Violin (both high-frequency)

### Evaluation Metrics

-    **Accuracy:** Overall correctness
-    **Precision:** True positives / (TP + FP)
-    **Recall:** True positives / (TP + FN)
-    **F1-Score:** Harmonic mean of precision & recall
-    **Confusion Matrix:** Per-class performance

---

## Slide 11: Demo Interface / Visualizations

### Interactive Web Interface

**Features:**

-    File upload (WAV/MP3)
-    Real-time audio processing
-    Step-by-step DSP visualization

### Visualization Pipeline

1. **Raw Waveform** - Time domain signal
2. **Filtered Waveform** - After bandpass filter
3. **FFT Spectrum** - Frequency domain
4. **Spectrogram** - Time-frequency representation
5. **MFCC Features** - Timbral characteristics
6. **Feature Vector** - 48-D numerical representation

**Additional Features:**

-    Predicted instrument with confidence scores
-    Probability distribution across classes
-    Professional gradient UI design

---

## Slide 12: Conclusion & Future Work

### Key Contributions

✅ Complete DSP pipeline implementation  
✅ Effective feature engineering (48-D vector)  
✅ Robust classification (85-90% accuracy)  
✅ Educational visualization interface  
✅ Open-source, well-documented code

### Current Limitations

-    Single-label classification only
-    Fixed 3-second duration
-    Limited dataset size
-    No temporal modeling

### Future Improvements

**Short-term:**

-    Data augmentation
-    Feature selection
-    Hyperparameter tuning

**Medium-term:**

-    Multi-label classification
-    Deep learning integration
-    Real-time processing

**Long-term:**

-    Hybrid models (traditional + deep learning)
-    Temporal modeling (RNNs/LSTMs)
-    Production deployment (REST API, cloud)

---

## Slide 13: Q&A / Thank You

# Thank You!

## Questions?

**Project Repository:** Available on GitHub  
**Documentation:** Complete technical reports (EN/VI)  
**Demo:** Interactive web interface at localhost:8501

**Contact:** For questions and collaboration

---

**End of Presentation**
