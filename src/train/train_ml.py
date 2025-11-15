# train_ml.py (moved to train/train_ml.py)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from src.data.dataset import list_audio_files, load_audio, label2id, id2label
from src.dsp.dsp import design_bandpass_fir, apply_filter
from src.dsp.features import build_feature_vector
from src.config.config import SAMPLE_RATE, AUDIO_DURATION, SVM_MODEL_PATH, SCALER_PATH

def build_dataset():
    """
    1. Duyệt file audio
    2. Load + resample + cắt/pad
    3. Lọc bandpass
    4. Trích feature vector
    5. Trả về X, y dạng numpy
    """
    files = list_audio_files()
    if len(files) == 0:
        raise RuntimeError(
            "Không tìm thấy file audio nào trong data/raw/<class>/*.wav. "
            "Vui lòng kiểm tra lại cấu trúc thư mục."
        )

    X_list = []
    y_list = []

    # Thiết kế filter 1 lần
    taps = design_bandpass_fir(sr=SAMPLE_RATE)

    for idx, (path, label_str) in enumerate(files):
        print(f"[PROCESS] {idx+1}/{len(files)}: {path} ({label_str})")
        try:
            # 1. Load audio
            signal, sr = load_audio(path, duration=AUDIO_DURATION, sr=SAMPLE_RATE)
            # 2. Lọc bandpass
            filtered = apply_filter(signal, taps)
            # 3. Trích feature
            feat_vec = build_feature_vector(filtered, sr=sr)
            X_list.append(feat_vec)
            y_list.append(label2id[label_str])
        except Exception as e:
            print(f"[ERROR] Lỗi xử lý file {path}: {e}")
            continue

    if not X_list:
        raise RuntimeError("Không có file audio nào hợp lệ để train.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def main():
    # 1. Xây dataset
    X, y = build_dataset()

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3. Chuẩn hóa feature
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Khởi tạo & train SVM
    clf = SVC(kernel="rbf", C=10.0, gamma="scale")
    clf.fit(X_train, y_train)

    # 5. Dự đoán & đánh giá
    y_pred = clf.predict(X_test)

    print("\n[RESULT] Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[id2label[i] for i in sorted(id2label.keys())],
        )
    )

    print("\n[RESULT] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 6. Lưu model + scaler để inference
    joblib.dump(clf, SVM_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n[SAVE] Model saved to:  {SVM_MODEL_PATH}")
    print(f"[SAVE] Scaler saved to: {SCALER_PATH}")

if __name__ == "__main__":
    main()
