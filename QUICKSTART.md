
# 🚀 QUICKSTART - INSTRUMENT CLASSIFICATION DSP PROJECT

## 1. SYSTEM REQUIREMENTS
- Python 3.10
- pip (Python package manager)
- (Recommended) Create a virtual environment: `python -m venv .venv`

## 2. INSTALL DEPENDENCIES
```bash
pip install -r requirements.txt
```

## 3. DATA PREPARATION
- Place audio files into corresponding folders in `data/raw/<Instrument>/`
- Each instrument is a subfolder (Drum, Flute, Guitar, Piano, Violin, Tambourine,...)

## 4. TRAIN THE MODEL
```bash
python -m src.train.train_ml
```
- Model and scaler will be saved in the `models/` folder

## 5. RUN THE WEB INTERFACE
```bash
streamlit run src/app_streamlit.py
```
- Access: http://localhost:8501
- Upload audio file, click Analyze to see results and DSP pipeline

## 6. MAIN FOLDER STRUCTURE
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

## 7. NOTES
- Do not push raw data to git (already ignored in `.gitignore`)
- You can push trained models to git if you want to share results
- If any library is missing, check `requirements.txt`