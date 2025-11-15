# ---

# English version below

# 🚀 QUICKSTART - INSTRUMENT CLASSIFICATION DSP PROJECT (ENGLISH)

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

---

# 🚀 QUICKSTART - INSTRUMENT CLASSIFICATION DSP PROJECT

## 1. YÊU CẦU HỆ THỐNG
- Python 3.10
- pip (Python package manager)
- (Khuyến nghị) Tạo virtual environment: `python -m venv .venv`

## 2. CÀI ĐẶT PHỤ THUỘC
```bash
pip install -r requirements.txt
```

## 3. CHUẨN BỊ DỮ LIỆU
- Đặt file audio vào các thư mục tương ứng trong `data/raw/<Instrument>/`
- Mỗi nhạc cụ là một thư mục con (Drum, Flute, Guitar, Piano, Violin, Tambourine,...)
- **Hoặc có thể lấy file mẫu trong thư mục `data/test/` để chạy thử nghiệm nhanh.**
## 3. DATA PREPARATION
- Place audio files into corresponding folders in `data/raw/<Instrument>/`
- Each instrument is a subfolder (Drum, Flute, Guitar, Piano, Violin, Tambourine,...)
- **Or you can use sample files in the `data/test/` folder for quick testing.**

## 4. TRAIN MODEL
```bash
python -m src.train.train_ml
```
- Model và scaler sẽ được lưu vào thư mục `models/`

## 5. CHẠY GIAO DIỆN WEB
```bash
streamlit run src/app_streamlit.py
```
- Truy cập: http://localhost:8501
- Upload file audio, nhấn Analyze để xem kết quả và pipeline DSP

## 6. CẤU TRÚC THƯ MỤC CHÍNH
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

## 7. LƯU Ý
- Không push dữ liệu gốc lên git (đã ignore trong `.gitignore`)
- Có thể push model đã train lên git nếu muốn chia sẻ kết quả
- Nếu thiếu thư viện, kiểm tra lại `requirements.txt`

---

**Chúc bạn demo thành công!** 🎵