import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import joblib
import pandas as pd

from dsp.dsp import design_bandpass_fir, apply_filter, compute_spectrogram
from dsp.features import build_feature_vector, extract_mfcc
from config.config import SAMPLE_RATE, AUDIO_DURATION, INSTRUMENT_CLASSES, SVM_MODEL_PATH, SCALER_PATH


# ========= Helper: chia audio thành nhiều đoạn 3s =========
def split_into_segments(signal: np.ndarray, sr: int, duration: float = AUDIO_DURATION):
    """
    Chia tín hiệu dài thành nhiều đoạn (duration giây).
    Đoạn cuối nếu ngắn hơn thì pad thêm 0 cho đủ.
    """
    segment_len = int(duration * sr)
    segments = []

    for start in range(0, len(signal), segment_len):
        end = start + segment_len
        chunk = signal[start:end]

        if len(chunk) == 0:
            continue
        # xử lý zero-padding cho đoạn cuối
        if len(chunk) < segment_len:
            chunk = np.pad(chunk, (0, segment_len - len(chunk)))

        segments.append(chunk.astype(np.float32))

    return segments


# ========= CSS =========
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="gradient-header">Instrument Classification Demo</div>', unsafe_allow_html=True)

# ========= Card upload =========
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 1. Upload audio file")
uploaded_file = st.file_uploader("Chọn file .wav hoặc .mp3", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    analyze = st.button("Analyze")

    if analyze:
        # ---- 1. Load full audio ----
        y, sr = librosa.load(uploaded_file, sr=None, mono=True)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        y = y.astype(np.float32)

        # ---- 2. Chia thành nhiều đoạn 3s ----
        segments = split_into_segments(y, sr, AUDIO_DURATION)

        if len(segments) == 0:
            st.error("File audio quá ngắn, không đủ dữ liệu để phân tích.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Đoạn 3s đầu để VISUALIZE pipeline
            signal = segments[0]

            # ---- 3. DSP: Bandpass Filter ----
            taps = design_bandpass_fir(sr=sr)
            filtered = apply_filter(signal, taps)

            # ---- 4. DSP: Spectrogram / MFCC cho đoạn đầu ----
            S_db = compute_spectrogram(filtered, sr=sr)
            mfcc = extract_mfcc(filtered, sr=sr)
            feat_vec_vis = build_feature_vector(filtered, sr=sr).reshape(1, -1)

            # ---- 5. Extract feature cho TẤT CẢ segment để phân loại ----
            all_feats = []
            for seg in segments:
                seg_filt = apply_filter(seg, taps)
                fv = build_feature_vector(seg_filt, sr=sr)
                all_feats.append(fv)

            all_feats = np.vstack(all_feats)  # (n_segments, n_features)

            # ---- 6. Load model + scaler, scale và predict ----
            clf = joblib.load(SVM_MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            feats_scaled = scaler.transform(all_feats)

            probas = None

            # Nếu SVM có predict_proba → soft voting theo xác suất trung bình
            if hasattr(clf, "predict_proba"):
                probas_segments = clf.predict_proba(feats_scaled)          # (n_seg, n_class)
                probas = probas_segments.mean(axis=0)                      # trung bình theo segment
                pred_idx = int(np.argmax(probas))
            else:
                # Nếu không có proba: dùng decision_function rồi softmax
                if hasattr(clf, "decision_function"):
                    dec = clf.decision_function(feats_scaled)              # (n_seg, n_class) hoặc (n_seg,)
                    dec = np.atleast_2d(dec)
                    dec_mean = dec.mean(axis=0)
                    exp_dec = np.exp(dec_mean - np.max(dec_mean))
                    probas = exp_dec / exp_dec.sum()
                    pred_idx = int(np.argmax(probas))
                else:
                    # Fallback: majority vote theo hard prediction
                    preds = clf.predict(feats_scaled)
                    values, counts = np.unique(preds, return_counts=True)
                    pred_idx = int(values[np.argmax(counts)])
                    probas = None

            pred_label = INSTRUMENT_CLASSES[pred_idx]

            # ---- 7. Card: Prediction ----
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 2. Model Prediction")
            st.success(f"Predicted instrument: {pred_label}")

            if probas is not None:
                st.markdown("#### Tỉ lệ dự đoán từng nhạc cụ (trung bình trên tất cả các đoạn 3s):")
                for cls, p in zip(INSTRUMENT_CLASSES, probas):
                    percent = float(p) * 100
                    st.write(f"{cls}: {percent:.2f}%")
                    st.progress(percent / 100)

            st.markdown('</div>', unsafe_allow_html=True)

            # ---- 8. Card: DSP Pipeline (cho đoạn 3s đầu tiên) ----
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 3. DSP Pipeline Visualization")

            # 1. Waveform (Raw Audio)
            st.subheader("1. Waveform (Raw Audio)")
            st.write("Biểu diễn tín hiệu âm thanh (3 giây đầu) theo miền thời gian.")
            st.write(f"Shape: {signal.shape}, Sample rate: {sr}")
            fig, ax = plt.subplots()
            librosa.display.waveshow(signal, sr=sr, ax=ax)
            st.pyplot(fig)

            # 2. Waveform After Bandpass Filter
            st.subheader("2. Waveform After Bandpass Filter")
            st.write("Tín hiệu sau khi lọc thông dải, loại bỏ nhiễu ngoài dải tần số quan tâm (3 giây đầu).")
            st.write(f"Shape: {filtered.shape}, Sample rate: {sr}")
            fig2, ax2 = plt.subplots()
            librosa.display.waveshow(filtered, sr=sr, ax=ax2)
            st.pyplot(fig2)

            # 3. Fourier Transform (Frequency Domain)
            st.subheader("3. Fourier Transform (Frequency Domain)")
            st.write("Chuyển đổi tín hiệu đã lọc sang miền tần số bằng FFT (Fast Fourier Transform).")
            fft = np.fft.fft(filtered)
            fft_freq = np.fft.fftfreq(len(filtered), d=1 / sr)
            magnitude = np.abs(fft)
            half = len(filtered) // 2
            fig_fft, ax_fft = plt.subplots()
            ax_fft.plot(fft_freq[:half], magnitude[:half], color='#00FFCC')
            ax_fft.set_xlabel('Frequency (Hz)')
            ax_fft.set_ylabel('Magnitude')
            ax_fft.set_title('Frequency Spectrum (FFT)')
            st.pyplot(fig_fft)

            # Hiển thị ma trận FFT (frequency & magnitude)
            st.markdown('**FFT Matrix (Frequency & Magnitude, first 512 values):**')
            fft_df = pd.DataFrame({
                'Frequency (Hz)': fft_freq[:min(512, half)],
                'Magnitude': magnitude[:min(512, half)]
            })
            st.dataframe(fft_df.style.format(precision=4), height=300, use_container_width=True)

            # 4. Spectrogram (dB)
            st.subheader("4. Spectrogram (dB)")
            st.write("Phổ tần số theo thời gian (3 giây đầu, sau lọc).")
            st.write(f"Spectrogram shape: {S_db.shape}")
            fig3, ax3 = plt.subplots()
            img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="log", ax=ax3)
            fig3.colorbar(img, ax=ax3, format="%+2.0f dB")
            st.pyplot(fig3)

            # 5. MFCC Features
            st.subheader("5. MFCC Features")
            st.write("Đặc trưng MFCC, thường dùng cho nhận diện âm thanh (3 giây đầu).")
            st.write(f"MFCC shape: {mfcc.shape}")
            fig4, ax4 = plt.subplots()
            img2 = librosa.display.specshow(mfcc, x_axis="time", ax=ax4)
            fig4.colorbar(img2, ax=ax4)
            st.pyplot(fig4)

            # 6. Feature Vector (for Model)
            st.subheader("6. Feature Vector (for Model)")
            st.write("Vector đặc trưng tổng hợp (từ 3 giây đầu), đầu vào cho mô hình phân loại.")
            st.write(f"Feature vector shape: {feat_vec_vis.shape}")
            feat_df = pd.DataFrame(
                feat_vec_vis,
                columns=[f"f{i + 1}" for i in range(feat_vec_vis.shape[1])]
            )
            st.dataframe(
                feat_df.style.format(precision=4).set_properties(
                    **{'background-color': '#1a2b2e', 'color': 'white'}
                ),
                height=120,
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('</div>', unsafe_allow_html=True)
