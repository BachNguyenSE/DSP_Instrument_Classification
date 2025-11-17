
# 🎵 Instrument Classification DSP Project

## Description
This project uses Digital Signal Processing (DSP) techniques combined with Machine Learning to classify musical instruments from audio files. The interactive web interface is built with Streamlit, allowing users to upload files, analyze the DSP pipeline, and view instrument classification results.

## Key Features
- Audio preprocessing and filtering
- Time domain → frequency domain conversion (FFT, Spectrogram)
- Feature extraction (MFCC, Spectral features)
- Instrument classification using SVM
- Beautiful, user-friendly web interface with step-by-step DSP visualization

## Quick Setup
```bash
# 1. Clone the repository
# 2. (Recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# 3. Install dependencies
pip install -r requirements.txt
```

## Data Preparation
- Link dataset: https://www.kaggle.com/datasets/abdulvahap/music-instrunment-sounds-for-classification?resource=download
- Place audio files into subfolders under `data/raw/<Instrument>/`
- Each instrument should have its own folder: Drum, Flute, Guitar, Piano, Violin, Tambourine, ...

## Train the Model
```bash
python -m src.train.train_ml
```

## Run the Web Interface
```bash
streamlit run src/app_streamlit.py
```
- Access at: http://localhost:8501

## Project Structure
```
├── data/
│   └── raw/
│       ├── Drum/
│       ├── Flute/
│       ├── Guitar/
│       ├── Piano/
│       ├── Violin/
│       └── Tambourine/
├── models/
├── src/
│   ├── config/
│   ├── data/
│   ├── dsp/
│   ├── train/
│   ├── inference/
│   └── app_streamlit.py
├── requirements.txt
├── PROJECT_OVERVIEW.md
├── QUICKSTART.md
└── .gitignore
```

## References
- [Librosa](https://librosa.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
- [DSP Guide](https://www.dspguide.com/)

## Contact
- Digital Signal Processing Project - University

---

**See also:**
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
- [QUICKSTART.md](QUICKSTART.md)