# Phân Loại Nhạc Cụ Dựa Trên Xử Lý Tín Hiệu Số (DSP)

## Tổng Hợp Lý Thuyết và Kiến Thức DSP Trong Dự Án

---

## Mục Lục

1. [Tổng Quan Dự Án](#tổng-quan-dự-án)
2. [Cơ Sở Lý Thuyết Xử Lý Tín Hiệu Số](#cơ-sở-lý-thuyết-xử-lý-tín-hiệu-số)
3. [Tiền Xử Lý Âm Thanh](#tiền-xử-lý-âm-thanh)
4. [Lọc Số (Digital Filtering)](#lọc-số-digital-filtering)
5. [Phân Tích Miền Tần Số](#phân-tích-miền-tần-số)
6. [Trích Xuất Đặc Trưng](#trích-xuất-đặc-trưng)
7. [Phân Loại bằng Máy Học](#phân-loại-bằng-máy-học)
8. [Quy Trình Hệ Thống](#quy-trình-hệ-thống)

---

## Tổng Quan Dự Án

Dự án này phát triển một **hệ thống phân loại nhạc cụ dựa trên Xử Lý Tín Hiệu Số (DSP)** để nhận diện các nhạc cụ (Drum, Flute, Guitar, Piano, Violin, Tambourine) từ các bản ghi âm. Hệ thống phân tích các đặc trưng âm học độc đáo của từng nhạc cụ thông qua các kỹ thuật xử lý tín hiệu và máy học.

**Các Thành Phần Chính:**

-    Tiền xử lý và lọc âm thanh
-    Biến đổi miền tần số
-    Trích xuất đặc trưng (MFCC, đặc trưng phổ)
-    Phân loại bằng Support Vector Machine (SVM)

---

## Cơ Sở Lý Thuyết Xử Lý Tín Hiệu Số

### DSP Là Gì?

**Xử Lý Tín Hiệu Số (Digital Signal Processing - DSP)** là việc xử lý toán học các tín hiệu rời rạc theo thời gian để trích xuất thông tin, lọc nhiễu, hoặc biến đổi tín hiệu cho mục đích phân tích. Khác với xử lý tương tự, DSP hoạt động trên các mẫu dữ liệu đã được số hóa.

### Khái Niệm Cơ Bản

#### 1. **Lấy Mẫu (Sampling)**

-    **Định nghĩa**: Chuyển đổi tín hiệu tương tự liên tục thành các mẫu rời rạc theo thời gian
-    **Tần Số Lấy Mẫu (Fs)**: Số lượng mẫu mỗi giây (Hz)
     -    Dự án sử dụng: **16,000 Hz (16 kHz)**
     -    Tần số lấy mẫu cao hơn nắm bắt nhiều thông tin tần số hơn nhưng yêu cầu bộ nhớ lớn hơn

#### 2. **Định Lý Nyquist-Shannon**

-    **Quy Tắc Quan Trọng**: Để biểu diễn chính xác một tín hiệu, tần số lấy mẫu phải ít nhất **gấp đôi tần số cao nhất** trong tín hiệu
-    **Tần Số Nyquist**: `Fs/2` = tần số biểu diễn tối đa
     -    Với 16 kHz: Tần số Nyquist = **8 kHz**
-    **Aliasing (Nếp Gấp)**: Nếu tần số lấy mẫu quá thấp, tần số cao sẽ xuất hiện như tần số thấp (méo tín hiệu)

#### 3. **Lượng Tử Hóa (Quantization)**

-    Chuyển đổi các giá trị biên độ liên tục thành các mức rời rạc
-    Ảnh hưởng đến chất lượng tín hiệu và dải động

---

## Tiền Xử Lý Âm Thanh

### 1. **Lấy Lại Mẫu (Resampling)**

**Lý Thuyết:**

-    Các file âm thanh có thể có tần số lấy mẫu khác nhau (22.05 kHz, 44.1 kHz, 48 kHz, v.v.)
-    **Chuẩn hóa** về một tần số cố định (16 kHz) đảm bảo xử lý nhất quán
-    Sử dụng thuật toán nội suy/giảm mẫu để thay đổi tần số lấy mẫu mà vẫn bảo toàn nội dung tần số

**Triển Khai:**

```python
y = librosa.resample(y, orig_sr=orig_sr, target_sr=16000)
```

**Tại Sao 16 kHz?**

-    Hầu hết đặc trưng nhạc cụ nằm dưới 8 kHz
-    Giảm chi phí tính toán
-    Duy trì độ phân giải tần số đủ cho phân loại

### 2. **Chuyển Đổi Mono**

**Lý Thuyết:**

-    Âm thanh stereo có kênh trái và phải
-    Đối với phân loại nhạc cụ, thường sử dụng **mono** (một kênh)
-    Chuyển đổi mono: lấy trung bình của kênh trái/phải hoặc chọn một kênh
-    Giảm chiều dữ liệu nhưng vẫn bảo toàn thông tin cần thiết

**Triển Khai:**

```python
y, sr = librosa.load(path, sr=None, mono=True)
```

### 3. **Chuẩn Hóa Độ Dài**

**Lý Thuyết:**

-    Các file âm thanh có độ dài khác nhau
-    Máy học yêu cầu vector đặc trưng có **độ dài cố định**
-    **Cắt ngắn**: Cắt âm thanh dài hơn về độ dài cố định (ví dụ: 3 giây)
-    **Zero-padding**: Kéo dài âm thanh ngắn hơn bằng cách thêm số 0

**Triển Khai:**

```python
target_len = int(3.0 * sample_rate)  # 3 giây
if len(y) > target_len:
    y = y[:target_len]  # Cắt ngắn
elif len(y) < target_len:
    y = np.pad(y, (0, target_len - len(y)))  # Thêm số 0
```

---

## Lọc Số (Digital Filtering)

### Lý Thuyết Bộ Lọc Số

**Mục Đích:** Loại bỏ các thành phần tần số không mong muốn hoặc tăng cường dải tần số mong muốn

### Bộ Lọc FIR Dải Thông (FIR Bandpass Filter)

Dự án sử dụng **bộ lọc FIR dải thông** với dải tần số **100 Hz - 8,000 Hz**.

#### 1. **Đặc Tính Bộ Lọc FIR**

-    **Đáp Ứng Xung**: Thời lượng hữu hạn (ổn định sau một số mẫu hữu hạn)
-    **Luôn Ổn Định**: Không có phản hồi, không có cực ngoài vòng tròn đơn vị
-    **Pha Tuyến Tính**: Bảo toàn hình dạng tín hiệu, không méo pha
-    **Dễ Thiết Kế**: Sử dụng phương pháp cửa sổ

#### 2. **Thiết Kế Bộ Lọc Dải Thông**

**Mục Đích:**

-    **Cắt Thấp (100 Hz)**: Loại bỏ offset DC, nhiễu tần số thấp, tiếng ồn
-    **Cắt Cao (8,000 Hz)**: Loại bỏ nhiễu tần số cao, bảo toàn tín hiệu trong giới hạn Nyquist

**Phương Pháp Thiết Kế: FIR Cửa Sổ (Hamming/Rectangular)**

```python
taps = firwin(
    numtaps=101,           # Bậc bộ lọc (số hệ số)
    cutoff=[100, 8000],    # Dải tần số [thấp, cao] Hz
    pass_zero=False,       # Dải thông (không phải thông thấp/thông cao)
    fs=16000               # Tần số lấy mẫu
)
```

**Bậc Bộ Lọc (numtaps):**

-    Bậc cao hơn = độ cắt sắc hơn, chọn lọc tần số tốt hơn
-    Đánh đổi: Tính toán nhiều hơn, độ trễ dài hơn

#### 3. **Triển Khai Bộ Lọc**

**Lọc dựa trên tích chập:**

```python
filtered = lfilter(taps, 1.0, signal)
```

-    **Tích Chập**: Thao tác cửa sổ trượt
-    Mỗi mẫu đầu ra = tổng có trọng số của các mẫu đầu vào
-    Hệ số (`taps`) xác định đáp ứng tần số

#### 4. **Đáp Ứng Tần Số**

-    **Dải Thông**: Tần số đi qua (100-8000 Hz)
-    **Dải Chặn**: Tần số bị suy giảm
-    **Dải Chuyển Tiếp**: Vùng giữa dải thông và dải chặn

---

## Phân Tích Miền Tần Số

### Từ Miền Thời Gian Đến Miền Tần Số

**Miền Thời Gian**: Biên độ tín hiệu theo thời gian (dạng sóng)  
**Miền Tần Số**: Biên độ tín hiệu theo tần số (phổ)

### 1. **Biến Đổi Fourier Rời Rạc (DFT)**

**Lý Thuyết:**

-    Phân tích tín hiệu thành các thành phần tần số cấu thành
-    Biểu diễn tín hiệu dưới dạng tổng của các sóng sin ở tần số khác nhau
-    Tiết lộ tần số nào có mặt và biên độ của chúng

**Công Thức Toán Học:**

```
X(k) = Σ x(n) · e^(-j2πkn/N)
```

trong đó:

-    `x(n)` = các mẫu miền thời gian
-    `X(k)` = các hệ số miền tần số
-    `N` = số mẫu
-    `k` = chỉ số bin tần số

### 2. **Biến Đổi Fourier Nhanh (FFT)**

**Triển Khai:**

-    **FFT** là thuật toán hiệu quả để tính DFT
-    Giảm độ phức tạp từ O(N²) xuống O(N log N)
-    Sử dụng phương pháp chia để trị

**Triển Khai Dự Án:**

```python
Y = np.fft.fft(signal)              # Tính FFT
freq = np.fft.fftfreq(N, d=1/sr)    # Các bin tần số
magnitude = np.abs(Y[:N//2])        # Lấy nửa dương (Nyquist)
```

**Điểm Quan Trọng:**

-    FFT trả về số phức: `magnitude` và `phase`
-    Chỉ cần tần số dương (tần số âm đối xứng)
-    Độ phân giải tần số: `Δf = Fs / N`

### 3. **Biến Đổi Fourier Thời Gian Ngắn (STFT)**

**Vấn Đề Với FFT:**

-    FFT giả định tín hiệu là tĩnh (tính chất không thay đổi theo thời gian)
-    Tín hiệu âm nhạc là **không tĩnh** (nội dung tần số thay đổi)

**Giải Pháp: STFT**

-    Chia tín hiệu thành các cửa sổ ngắn chồng lên nhau
-    Tính FFT trên mỗi cửa sổ
-    Kết quả: **Biểu diễn thời gian-tần số** (spectrogram)

**Tham Số:**

-    **n_fft = 1024**: Kích thước cửa sổ cho FFT (độ phân giải tần số)
-    **hop_length = 512**: Bước nhảy giữa các cửa sổ (độ phân giải thời gian)

**Đánh Đổi:**

-    Cửa sổ lớn hơn = độ phân giải tần số tốt hơn, độ phân giải thời gian kém hơn
-    Cửa sổ nhỏ hơn = độ phân giải thời gian tốt hơn, độ phân giải tần số kém hơn

**Triển Khai:**

```python
stft = librosa.stft(signal, n_fft=1024, hop_length=512)
S_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
```

**Spectrogram:**

-    Biểu diễn 2D: Thời gian (trục x) vs. Tần số (trục y) vs. Biên độ (màu sắc)
-    Trực quan hóa cách nội dung tần số thay đổi theo thời gian

---

## Trích Xuất Đặc Trưng

Trích xuất đặc trưng chuyển đổi tín hiệu âm thanh thô thành các vector số học nắm bắt các đặc điểm đặc trưng của từng nhạc cụ.

### 1. **MFCC (Mel-Frequency Cepstral Coefficients)**

**Tại Sao MFCC?**

-    Nắm bắt đặc trưng âm sắc (chất lượng âm thanh, kết cấu)
-    Dựa trên nhận thức thính giác của con người
-    Biểu diễn gọn nhẹ (thường 13-20 hệ số)
-    Hiệu quả cho phân loại nhạc cụ

#### Quy Trình Tính Toán MFCC:

**Bước 1: Pre-emphasis (Nhấn Mạnh Trước)**

-    Tăng cường tần số cao để nhấn mạnh chi tiết phổ quan trọng
-    `y[n] = x[n] - α·x[n-1]` (thường α = 0.97)

**Bước 2: Cửa Sổ (Windowing)**

-    Chia tín hiệu thành các khung (thường 25 ms, chồng 50%)
-    Áp dụng hàm cửa sổ (Hamming/Hann) để giảm rò rỉ phổ

**Bước 3: FFT**

-    Tính phổ công suất cho mỗi khung

**Bước 4: Mel-Scale Filterbank**

-    **Thang Mel**: Thang nhận thức độ cao (con người nhận thức độ cao theo logarit)
-    Áp dụng bộ lọc tam giác trên thang Mel (thường 26-40 bộ lọc)
-    Chuyển đổi tần số (Hz) sang Mel: `mel = 2595·log₁₀(1 + f/700)`

**Bước 5: Logarithm**

-    Lấy log của năng lượng filterbank
-    Mô phỏng nhận thức độ lớn của con người (logarit)

**Bước 6: DCT (Discrete Cosine Transform)**

-    Nén đầu ra filterbank thành ít hệ số hơn
-    13-20 hệ số đầu tiên (MFCCs) nắm bắt hầu hết thông tin
-    Hệ số cao hơn biểu diễn chi tiết phổ mịn (thường là nhiễu)

**Triển Khai Dự Án:**

```python
mfcc = librosa.feature.mfcc(y=signal, sr=16000, n_mfcc=20)
mfcc_mean = mfcc.mean(axis=1)  # Trung bình theo thời gian
mfcc_std = mfcc.std(axis=1)    # Độ lệch chuẩn (nắm bắt biến thiên)
```

**Vector Đặc Trưng:**

-    20 MFCC trung bình + 20 MFCC độ lệch chuẩn = **40 đặc trưng MFCC**

### 2. **Đặc Trưng Phổ**

#### A. **Tâm Phổ (Spectral Centroid)**

**Định Nghĩa:** Độ "sáng" của âm thanh - trung bình có trọng số của tần số  
**Công Thức:** `centroid = Σ(f(k) · magnitude(k)) / Σ(magnitude(k))`

**Giải Thích:**

-    **Tâm phổ cao** (ví dụ: flute, violin): Âm thanh sáng, sắc
-    **Tâm phổ thấp** (ví dụ: piano, bass): Âm thanh tối, ấm

**Đặc Trưng Nhạc Cụ:**

-    Violin: ~2000-4000 Hz
-    Sáo (flute): ~2000-5000 Hz
-    Guitar: ~1000-3000 Hz
-    Piano: ~500-2000 Hz
-    Drum/Tambourine: Dải tần số rộng (nhạc cụ gõ)

#### B. **Băng Thông Phổ (Spectral Bandwidth)**

**Định Nghĩa:** Độ lan truyền của phổ quanh tâm phổ (đo lường sự tập trung phổ)  
**Công Thức:** `bandwidth = √(Σ((f(k) - centroid)² · magnitude(k)) / Σ(magnitude(k)))`

**Giải Thích:**

-    **Băng thông hẹp**: Nội dung tần số tập trung (ví dụ: âm thanh thuần)
-    **Băng thông rộng**: Nội dung tần số lan truyền (ví dụ: âm thanh giống nhiễu, nhạc cụ gõ)

#### C. **Độ Rơi Phổ (Spectral Rolloff)**

**Định Nghĩa:** Tần số dưới đó một phần trăm nhất định (thường 85%) năng lượng phổ được chứa

**Giải Thích:**

-    Đo lường hình dạng phổ và phân bố năng lượng
-    Độ rơi thấp: Năng lượng tập trung ở tần số thấp
-    Độ rơi cao: Năng lượng lan truyền đến tần số cao hơn

#### D. **Tỷ Lệ Giao Không (Zero Crossing Rate - ZCR)**

**Định Nghĩa:** Tỷ lệ tín hiệu vượt qua biên độ không mỗi đơn vị thời gian  
**Công Thức:** Đếm số lần giao không / tổng số mẫu

**Giải Thích:**

-    **ZCR cao**: Nội dung nhiễu, tần số cao (ví dụ: chũm chọe, phụ âm)
-    **ZCR thấp**: Nội dung mịn, tần số thấp (ví dụ: nguyên âm, âm thanh duy trì)
-    Hữu ích để phân biệt nhạc cụ gõ với nhạc cụ duy trì

**Triển Khai Dự Án:**

```python
centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y=signal)

# Tổng hợp: trung bình và độ lệch chuẩn qua các khung thời gian
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

**Tổng Đặc Trưng Phổ: 8** (4 đặc trưng × 2 thống kê)

### 3. **Vector Đặc Trưng Hoàn Chỉnh**

**Kích Thước Vector Đặc Trưng Cuối Cùng:**

-    40 đặc trưng MFCC (20 trung bình + 20 độ lệch chuẩn)
-    8 đặc trưng phổ (4 đặc trưng × 2 thống kê)
-    **Tổng: 48 đặc trưng cho mỗi mẫu âm thanh**

---

## Phân Loại bằng Máy Học

### Chuẩn Hóa Đặc Trưng

**StandardScaler (Chuẩn hóa Z-score):**

```python
z = (x - μ) / σ
```

trong đó:

-    `μ` = trung bình của đặc trưng
-    `σ` = độ lệch chuẩn của đặc trưng

**Tại Sao Chuẩn Hóa?**

-    Các đặc trưng có thang đo khác nhau (MFCC vs. tâm phổ)
-    Ngăn các đặc trưng có độ lớn lớn hơn chi phối
-    Cải thiện sự hội tụ và độ chính xác của SVM

### Support Vector Machine (SVM)

#### Lý Thuyết

**Khái Niệm:** Tìm siêu phẳng tối ưu phân tách các lớp với biên cực đại

**Thành Phần Chính:**

1. **Support Vectors**: Các điểm dữ liệu gần nhất với ranh giới quyết định
2. **Margin**: Khoảng cách giữa siêu phẳng và các điểm gần nhất
3. **Kernel Trick**: Ánh xạ dữ liệu sang không gian chiều cao hơn để phân tách phi tuyến

#### Kernel RBF (Radial Basis Function)

**Công Thức:** `K(x₁, x₂) = exp(-γ·||x₁ - x₂||²)`

**Tham Số:**

-    **C = 10.0**: Hình phạt cho phân loại sai
     -    C cao hơn: Nghiêm ngặt hơn, có thể quá khớp
     -    C thấp hơn: Linh hoạt hơn, có thể chưa khớp
-    **gamma = "scale"**: Hệ số kernel
     -    Điều khiển ảnh hưởng của từng mẫu riêng lẻ
     -    "scale" = tự động điều chỉnh dựa trên phương sai đặc trưng

**Tại Sao Kernel RBF?**

-    Xử lý ranh giới quyết định phi tuyến
-    Hiệu quả cho không gian đặc trưng chiều cao
-    Hiệu suất tốt cho các tác vụ phân loại âm thanh

**Cấu Hình Dự Án:**

```python
clf = SVC(kernel="rbf", C=10.0, gamma="scale")
```

---

## Quy Trình Hệ Thống

### Quy Trình Huấn Luyện

```
File Âm Thanh (WAV)
    ↓
1. Tải & Lấy Lại Mẫu (→ 16 kHz, mono)
    ↓
2. Chuẩn Hóa Độ Dài (cắt/thêm → 3s)
    ↓
3. Bộ Lọc FIR Dải Thông (100-8000 Hz)
    ↓
4. Trích Xuất Đặc Trưng
    ├─ MFCC (20 hệ số × 2 thống kê = 40)
    └─ Đặc Trưng Phổ (4 đặc trưng × 2 thống kê = 8)
    ↓
5. Vector Đặc Trưng (48 chiều)
    ↓
6. StandardScaler (chuẩn hóa)
    ↓
7. Huấn Luyện SVM (kernel RBF)
    ↓
Model + Scaler Đã Lưu
```

### Quy Trình Suy Luận

```
File Âm Thanh Test
    ↓
1. Tải & Tiền Xử Lý (giống huấn luyện)
    ↓
2. Lọc & Trích Xuất Đặc Trưng
    ↓
3. Chuẩn Hóa (sử dụng scaler đã lưu)
    ↓
4. Dự Đoán (sử dụng model SVM đã lưu)
    ↓
Kết Quả Phân Loại Nhạc Cụ
```

### Tại Sao Quy Trình Này Hoạt Động

1. **Tiền Xử Lý**: Chuẩn hóa đầu vào → phân tích nhất quán
2. **Lọc**: Loại bỏ nhiễu → đặc trưng sạch hơn
3. **FFT/STFT**: Tiết lộ nội dung tần số → âm sắc nhạc cụ
4. **MFCC**: Nắm bắt đặc trưng âm sắc → liên quan đến nhận thức
5. **Đặc Trưng Phổ**: Nắm bắt độ sáng, băng thông → đặc trưng phân biệt
6. **SVM**: Học ranh giới quyết định → phân loại mạnh mẽ

---

## Tóm Tắt Kiến Thức DSP

### Kỹ Thuật DSP Cốt Lõi

1. **Lấy Mẫu & Lượng Tử Hóa**: Chuyển đổi Tương tự → Số
2. **Định Lý Nyquist**: Yêu cầu tần số lấy mẫu
3. **Lọc Số**: Lọc FIR dải thông
4. **Phân Tích Miền Tần Số**: FFT, STFT, Spectrogram
5. **Nhận Thức Thang Mel**: Tính toán MFCC
6. **Kỹ Thuật Đặc Trưng**: Tổng hợp thống kê
7. **Chuẩn Hóa**: Chuẩn hóa đặc trưng
8. **Máy Học**: Phân loại SVM

### Nền Tảng Toán Học

-    **Đại Số Tuyến Tính**: Thao tác vector, biến đổi ma trận
-    **Xử Lý Tín Hiệu**: Tích chập, biến đổi Fourier, lọc
-    **Thống Kê**: Trung bình, độ lệch chuẩn, chuẩn hóa
-    **Tối Ưu Hóa**: Tối đa hóa biên SVM

### Ứng Dụng Thực Tế

-    Truy xuất thông tin âm nhạc
-    Hệ thống phân loại âm thanh
-    Nhận dạng giọng nói
-    Nâng cao chất lượng âm thanh
-    Phân tích âm học

---

## Kết Luận

Dự án này minh họa cách các kỹ thuật **Xử Lý Tín Hiệu Số** có thể trích xuất các đặc trưng có ý nghĩa từ tín hiệu âm thanh để phân loại nhạc cụ. Bằng cách kết hợp:

-    **Tiền Xử Lý** (lấy lại mẫu, lọc)
-    **Phân Tích Tần Số** (FFT, STFT)
-    **Đặc Trưng Nhận Thức** (MFCC)
-    **Đặc Trưng Thống Kê** (tính chất phổ)
-    **Máy Học** (SVM)

Chúng ta tạo ra một hệ thống mạnh mẽ có khả năng phân biệt giữa các nhạc cụ khác nhau dựa trên "dấu vân tay" âm học độc đáo của chúng.

**Ý Tưởng Chính**: Mỗi nhạc cụ có một "dấu vân tay" riêng biệt trong miền tần số, có thể được nắm bắt và phân loại thông qua các kỹ thuật DSP phù hợp.

---

## Tài Liệu Tham Khảo

-    Oppenheim & Schafer: _Discrete-Time Signal Processing_
-    Rabiner & Schafer: _Digital Processing of Speech Signals_
-    Librosa Documentation: Thư viện phân tích âm thanh
-    Scikit-learn: Thư viện máy học

---

_Kết Thúc Bài Thuyết Trình_
