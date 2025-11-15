import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import joblib
from dsp.dsp import design_bandpass_fir, apply_filter, compute_spectrogram
from dsp.features import build_feature_vector, extract_mfcc
from config.config import SAMPLE_RATE, AUDIO_DURATION, INSTRUMENT_CLASSES, SVM_MODEL_PATH, SCALER_PATH

# CSS gradient nền mới + style chuyên nghiệp, toàn bộ chữ màu trắng
st.markdown(
    """
    <style>
    body, .stApp {
        background: linear-gradient(135deg, #0F2027 0%, #28623A 100%) !important;
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
        color: #FFFFFF !important;
    }
    .gradient-header {
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        color: #FFFFFF !important;
        margin-bottom: 1.5rem;
        letter-spacing: 2px;
        background: none !important;
        -webkit-background-clip: unset !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    .card {
        background: rgba(40,98,58,0.10);
        border-radius: 18px;
        box-shadow: 0 4px 24px 0 rgba(15,32,39,0.15);
        padding: 2rem 1.5rem 1.5rem 1.5rem;
        margin-bottom: 2rem;
        color: #FFFFFF !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #28623A, #0F2027);
        color: #fff;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 2rem;
        margin: 0.5rem 0 1.5rem 0;
        font-size: 1.1rem;
    }
    .stAudio {
        margin-bottom: 1.5rem;
    }
    .stSuccess {
        background: linear-gradient(90deg, #28623A 60%, #0F2027);
        color: #fff !important;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .stSubheader, .stMarkdown h3, .stMarkdown h2, .stMarkdown h1, .stMarkdown p, .stMarkdown ul, .stMarkdown li {
        color: #FFFFFF !important;
    }
    /* Ẩn menu và footer mặc định */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="gradient-header">Instrument Classification Demo</div>', unsafe_allow_html=True)

# Card upload luôn hiển thị
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 1. Upload audio file")
uploaded_file = st.file_uploader("Chọn file .wav hoặc .mp3", type=["wav", "mp3"])
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    analyze = st.button("Analyze")
    if analyze:
        # Xử lý và chỉ render các card khi có kết quả
        # Load audio
        y, sr = librosa.load(uploaded_file, sr=None, mono=True)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        target_len = int(AUDIO_DURATION * sr)
        if len(y) > target_len:
            y = y[:target_len]
        elif len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        signal = y.astype(np.float32)

        # DSP: Bandpass Filter
        taps = design_bandpass_fir(sr=sr)
        filtered = apply_filter(signal, taps)

        # DSP: Spectrogram
        S_db = compute_spectrogram(filtered, sr=sr)

        # Feature Extraction (MFCC + spectral)
        mfcc = extract_mfcc(filtered, sr=sr)
        feat_vec = build_feature_vector(filtered, sr=sr).reshape(1, -1)

        # Predict
        clf = joblib.load(SVM_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feat_scaled = scaler.transform(feat_vec)
        pred_idx = int(clf.predict(feat_scaled)[0])
        pred_label = INSTRUMENT_CLASSES[pred_idx]

        # Tính xác suất dự đoán cho từng class (nếu SVC hỗ trợ)
        probas = None
        if hasattr(clf, "predict_proba"):
            probas = clf.predict_proba(feat_scaled)[0]
        else:
            # Nếu không có predict_proba, dùng decision_function và chuẩn hóa
            if hasattr(clf, "decision_function"):
                dec = clf.decision_function(feat_scaled)[0]
                # Softmax cho tạm (không phải xác suất tuyệt đối)
                exp_dec = np.exp(dec - np.max(dec))
                probas = exp_dec / exp_dec.sum()
        # Card: Prediction
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 2. Model Prediction")
        st.success(f"Predicted instrument: {pred_label}")
        if probas is not None:
            st.markdown("#### Tỉ lệ dự đoán từng nhạc cụ:")
            for cls, p in zip(INSTRUMENT_CLASSES, probas):
                percent = float(p) * 100
                st.write(f"{cls}: {percent:.2f}%")
                st.progress(percent / 100)
        st.markdown('</div>', unsafe_allow_html=True)

        # Card: DSP Pipeline
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 3. DSP Pipeline Visualization")

        # 1. Waveform (Raw Audio)
        st.subheader("1. Waveform (Raw Audio)")
        st.write("Biểu diễn tín hiệu âm thanh gốc theo miền thời gian.")
        st.write(f"Shape: {signal.shape}, Sample rate: {sr}")
        fig, ax = plt.subplots()
        librosa.display.waveshow(signal, sr=sr, ax=ax)
        st.pyplot(fig)

        # 2. Waveform After Bandpass Filter
        st.subheader("2. Waveform After Bandpass Filter")
        st.write("Tín hiệu sau khi lọc thông dải, loại bỏ nhiễu ngoài dải tần số quan tâm.")
        st.write(f"Shape: {filtered.shape}, Sample rate: {sr}")
        fig2, ax2 = plt.subplots()
        librosa.display.waveshow(filtered, sr=sr, ax=ax2)
        st.pyplot(fig2)

        # 3. Fourier Transform (Frequency Domain)
        st.subheader("3. Fourier Transform (Frequency Domain)")
        st.write("Chuyển đổi tín hiệu đã lọc sang miền tần số bằng FFT (Fast Fourier Transform). Quan sát các thành phần tần số chính.")
        fft = np.fft.fft(filtered)
        fft_freq = np.fft.fftfreq(len(filtered), d=1/sr)
        magnitude = np.abs(fft)
        half = len(filtered) // 2
        fig_fft, ax_fft = plt.subplots()
        ax_fft.plot(fft_freq[:half], magnitude[:half], color='#00FFCC')
        ax_fft.set_xlabel('Frequency (Hz)')
        ax_fft.set_ylabel('Magnitude')
        ax_fft.set_title('Frequency Spectrum (FFT)')
        st.pyplot(fig_fft)

        # 4. Spectrogram (dB)
        st.subheader("4. Spectrogram (dB)")
        st.write("Phổ tần số theo thời gian (sau lọc), trực quan hóa sự thay đổi tần số trong tín hiệu.")
        st.write(f"Spectrogram shape: {S_db.shape}")
        fig3, ax3 = plt.subplots()
        img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="log", ax=ax3)
        fig3.colorbar(img, ax=ax3, format="%+2.0f dB")
        st.pyplot(fig3)

        # 5. MFCC Features
        st.subheader("5. MFCC Features")
        st.write("Đặc trưng MFCC, thường dùng cho nhận diện âm thanh.")
        st.write(f"MFCC shape: {mfcc.shape}")
        fig4, ax4 = plt.subplots()
        img2 = librosa.display.specshow(mfcc, x_axis="time", ax=ax4)
        fig4.colorbar(img2, ax=ax4)
        st.pyplot(fig4)

        # 6. Feature Vector (for Model)
        st.subheader("6. Feature Vector (for Model)")
        st.write("Vector đặc trưng tổng hợp, đầu vào cho mô hình phân loại.")
        st.write(f"Feature vector shape: {feat_vec.shape}")
        import pandas as pd
        feat_df = pd.DataFrame(feat_vec, columns=[f"f{i+1}" for i in range(feat_vec.shape[1])])
        st.dataframe(feat_df.style.format(precision=4).set_properties(**{'background-color': '#1a2b2e', 'color': 'white'}),
                 height=120, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('</div>', unsafe_allow_html=True)
