# src/config.py
from pathlib import Path

# Project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Sample rate dùng chung cho toàn project
SAMPLE_RATE: int = 16000

# Độ dài cố định mỗi đoạn audio (giây)
AUDIO_DURATION: float = 3.0  # 3s

# Số lượng hệ số MFCC
N_MFCC: int = 20

# Danh sách các class nhạc cụ
INSTRUMENT_CLASSES = [
    "organ",
    "guitar",
    "brass",
    "flute",
]

# Đường dẫn thư mục dữ liệu
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FEATURE_DIR: Path = DATA_DIR / "features"

# Thư mục model & test
MODEL_DIR: Path = PROJECT_ROOT / "models"
TEST_DIR: Path = PROJECT_ROOT / "data" / "test"

# File model + scaler
SVM_MODEL_PATH: Path = MODEL_DIR / "svm_instrument.joblib"
SCALER_PATH: Path = MODEL_DIR / "scaler.joblib"

# Tạo thư mục nếu chưa tồn tại
for d in [RAW_DATA_DIR, PROCESSED_DIR, FEATURE_DIR, MODEL_DIR, TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)
