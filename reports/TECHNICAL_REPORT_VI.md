# Hệ Thống Phân Loại Nhạc Cụ Dựa Trên Xử Lý Tín Hiệu Số

**Báo Cáo Kỹ Thuật Toàn Diện**

---

## Tóm Tắt

Báo cáo này trình bày một hệ thống hoàn chỉnh dựa trên Xử Lý Tín Hiệu Số (DSP) để phân loại tự động các nhạc cụ từ bản ghi âm. Hệ thống sử dụng kết hợp các kỹ thuật DSP truyền thống và máy học để nhận diện sáu lớp nhạc cụ khác nhau: Trống (Drum), Sáo (Flute), Guitar, Piano, Violin, và Tambourine. Phương pháp bao gồm tiền xử lý âm thanh thông qua lấy lại mẫu và lọc thông dải FIR, phân tích miền tần số sử dụng Biến Đổi Fourier Nhanh (FFT) và Biến Đổi Fourier Thời Gian Ngắn (STFT), trích xuất đặc trưng qua Hệ Số Cepstral Tần Số Mel (MFCC) và đặc trưng phổ, và phân loại sử dụng Máy Vector Hỗ Trợ (SVM) với kernel Hàm Cơ Sở Bán Kính (RBF). Hệ thống đạt độ chính xác huấn luyện khoảng 90-95% và độ chính xác kiểm tra 85-90% trên tập dữ liệu thử nghiệm. Giao diện web thân thiện được xây dựng bằng Streamlit cung cấp phân tích âm thanh thời gian thực và trực quan hóa toàn bộ pipeline DSP. Công trình này chứng minh việc áp dụng hiệu quả các nguyên lý DSP cơ bản trong giải quyết các vấn đề phân loại âm thanh thực tế, phù hợp cho truy xuất thông tin âm nhạc, phân tích âm thanh, và mục đích giáo dục.

**Từ Khóa:** Xử Lý Tín Hiệu Số, Phân Loại Âm Thanh, MFCC, SVM, Trích Xuất Đặc Trưng, Nhạc Cụ

---

## Mục Lục

1. [Giới Thiệu](#1-giới-thiệu)
2. [Tổng Quan Tài Liệu và Cơ Sở Lý Thuyết](#2-tổng-quan-tài-liệu-và-cơ-sở-lý-thuyết)
3. [Phương Pháp](#3-phương-pháp)
4. [Triển Khai](#4-triển-khai)
5. [Thiết Lập Thực Nghiệm](#5-thiết-lập-thực-nghiệm)
6. [Kết Quả và Thảo Luận](#6-kết-quả-và-thảo-luận)
7. [Kết Luận và Hướng Phát Triển](#7-kết-luận-và-hướng-phát-triển)
8. [Tài Liệu Tham Khảo](#8-tài-liệu-tham-khảo)

---

## 1. Giới Thiệu

### 1.1 Phát Biểu Vấn Đề

Phân loại tự động nhạc cụ từ bản ghi âm là một vấn đề cơ bản trong truy xuất thông tin âm nhạc và phân tích âm thanh. Thách thức nằm ở việc trích xuất các đặc trưng âm học có ý nghĩa có thể phân biệt giữa các nhạc cụ khác nhau, mỗi nhạc cụ sở hữu đặc điểm âm sắc, nội dung tần số và mẫu thời gian độc đáo. Các phương pháp truyền thống yêu cầu kỹ thuật đặc trưng thủ công và chuyên môn về lĩnh vực, trong khi các phương pháp học sâu hiện đại thường yêu cầu tập dữ liệu lớn và tài nguyên tính toán đáng kể.

### 1.2 Mục Tiêu

Các mục tiêu chính của dự án này là:

1. **Phát triển một pipeline DSP hoàn chỉnh** cho phân loại nhạc cụ thể hiện các khái niệm xử lý tín hiệu cơ bản bao gồm lấy mẫu, lọc, phân tích miền tần số, và trích xuất đặc trưng.

2. **Triển khai và đánh giá** một hệ thống phân loại máy học sử dụng Máy Vector Hỗ Trợ (SVM) với các đặc trưng được thiết kế cẩn thận trích xuất từ tín hiệu âm thanh.

3. **Tạo giao diện web tương tác** trực quan hóa từng bước của pipeline DSP, làm cho hệ thống dễ tiếp cận cho cả mục đích sử dụng thực tế và giáo dục.

4. **Đạt hiệu suất phân loại mạnh mẽ** trên sáu lớp nhạc cụ trong khi duy trì hiệu quả tính toán và khả năng giải thích.

### 1.3 Phạm Vi và Đóng Góp

Dự án này tập trung vào phân loại sáu nhạc cụ: Trống (Drum), Sáo (Flute), Guitar, Piano, Violin, và Tambourine. Hệ thống xử lý các file âm thanh có độ dài tối đa 3 giây, chuẩn hóa chúng về tần số lấy mẫu 16 kHz, và trích xuất vector đặc trưng 48 chiều để phân loại.

**Đóng Góp Chính:**

-    Triển khai pipeline DSP hoàn chỉnh, đầu cuối thể hiện các kỹ thuật xử lý tín hiệu cơ bản
-    Tích hợp các phương pháp DSP truyền thống (lọc, FFT, STFT) với máy học
-    Trích xuất đặc trưng toàn diện kết hợp MFCC và đặc trưng phổ
-    Hệ thống trực quan hóa tương tác giáo dục người dùng về khái niệm DSP
-    Triển khai mã nguồn mở phù hợp cho mục đích giáo dục và nghiên cứu

---

## 2. Tổng Quan Tài Liệu và Cơ Sở Lý Thuyết

### 2.1 Cơ Sở Lý Thuyết Xử Lý Tín Hiệu Số

Xử Lý Tín Hiệu Số (DSP) là việc thao tác toán học các tín hiệu rời rạc theo thời gian để trích xuất thông tin, lọc nhiễu, hoặc biến đổi tín hiệu cho mục đích phân tích. Khác với xử lý tương tự, DSP hoạt động trên các điểm dữ liệu đã được lấy mẫu, cho phép kiểm soát chính xác và khả năng tái tạo.

#### 2.1.1 Lấy Mẫu và Định Lý Nyquist-Shannon

Quá trình chuyển đổi tín hiệu tương tự liên tục theo thời gian thành các mẫu rời rạc được gọi là lấy mẫu. **Định Lý Lấy Mẫu Nyquist-Shannon** phát biểu rằng để biểu diễn chính xác một tín hiệu, tần số lấy mẫu phải ít nhất gấp đôi tần số cao nhất có trong tín hiệu. **Tần số Nyquist** được định nghĩa là `Fs/2`, trong đó `Fs` là tần số lấy mẫu. Đối với tần số lấy mẫu 16 kHz, tần số Nyquist là 8 kHz, nghĩa là các tần số lên đến 8 kHz có thể được biểu diễn chính xác.

**Aliasing (Nếp Gấp)** xảy ra khi tần số lấy mẫu quá thấp, khiến các tần số cao xuất hiện như tần số thấp, dẫn đến méo tín hiệu. Dự án này sử dụng tần số lấy mẫu 16 kHz, đủ cho phân loại nhạc cụ vì hầu hết đặc điểm nhạc cụ nằm dưới 8 kHz.

#### 2.1.2 Lọc Số

Bộ lọc số được sử dụng để loại bỏ các thành phần tần số không mong muốn hoặc tăng cường các dải tần số mong muốn. **Bộ Lọc Đáp Ứng Xung Hữu Hạn (FIR)** được đặc trưng bởi:

-    Đáp ứng xung thời lượng hữu hạn (ổn định sau một số mẫu hữu hạn)
-    Luôn ổn định (không có phản hồi, không có cực ngoài vòng tròn đơn vị)
-    Pha tuyến tính (bảo toàn hình dạng tín hiệu, không méo pha)
-    Dễ thiết kế hơn sử dụng phương pháp cửa sổ

Dự án này sử dụng bộ lọc thông dải FIR với dải tần số 100-8000 Hz để loại bỏ offset DC, nhiễu tần số thấp và các hiện tượng tần số cao trong khi bảo toàn nội dung tần số liên quan đến phân loại nhạc cụ.

#### 2.1.3 Phân Tích Miền Tần Số

**Biến Đổi Fourier** chuyển đổi tín hiệu từ miền thời gian sang miền tần số, tiết lộ các thành phần tần số có trong tín hiệu. **Biến Đổi Fourier Nhanh (FFT)** là một thuật toán hiệu quả tính toán Biến Đổi Fourier Rời Rạc (DFT) với độ phức tạp O(N log N) thay vì O(N²).

Đối với tín hiệu không tĩnh như âm nhạc, **Biến Đổi Fourier Thời Gian Ngắn (STFT)** chia tín hiệu thành các cửa sổ ngắn chồng lên nhau và tính FFT trên mỗi cửa sổ, tạo ra biểu diễn thời gian-tần số gọi là **spectrogram**.

### 2.2 Trích Xuất Đặc Trưng cho Phân Loại Âm Thanh

#### 2.2.1 Hệ Số Cepstral Tần Số Mel (MFCC)

Đặc trưng MFCC được sử dụng rộng rãi trong xử lý âm thanh và giọng nói vì chúng:

-    Nắm bắt đặc điểm âm sắc (chất lượng âm thanh, kết cấu)
-    Dựa trên nhận thức thính giác của con người (thang Mel)
-    Cung cấp biểu diễn gọn nhẹ (thường 13-20 hệ số)
-    Hiệu quả cho phân loại nhạc cụ

Quy trình tính toán MFCC bao gồm:

1. Pre-emphasis (tăng cường tần số cao)
2. Windowing (chia thành các khung)
3. FFT (tính phổ công suất)
4. Mel-scale filterbank (áp dụng bộ lọc tam giác)
5. Logarithm (mô phỏng nhận thức độ lớn của con người)
6. Biến Đổi Cosine Rời Rạc (DCT) để nén đặc trưng

#### 2.2.2 Đặc Trưng Phổ

Các đặc trưng phổ bổ sung cung cấp thông tin bổ sung:

-    **Tâm Phổ (Spectral Centroid)**: Trung bình có trọng số của tần số, chỉ ra "độ sáng"
-    **Băng Thông Phổ (Spectral Bandwidth)**: Độ lan truyền của phổ quanh tâm phổ
-    **Độ Rơi Phổ (Spectral Rolloff)**: Tần số dưới đó 85% năng lượng được chứa
-    **Tỷ Lệ Giao Không (Zero Crossing Rate - ZCR)**: Tỷ lệ thay đổi dấu, hữu ích để phân biệt nhạc cụ gõ với nhạc cụ duy trì

### 2.3 Máy Học cho Phân Loại Âm Thanh

Máy Vector Hỗ Trợ (SVM) là một bộ phân loại mạnh mẽ tìm siêu phẳng tối ưu để phân tách các lớp với biên cực đại. **Kernel RBF (Hàm Cơ Sở Bán Kính)** cho phép phân loại phi tuyến bằng cách ánh xạ dữ liệu sang không gian chiều cao hơn, làm cho nó phù hợp cho không gian đặc trưng âm thanh phức tạp.

---

## 3. Phương Pháp

### 3.1 Kiến Trúc Hệ Thống

Hệ thống tuân theo kiến trúc mô-đun với sự phân tách rõ ràng các mối quan tâm:

```
┌──────────────────────────────────────────────┐
│    Giao Diện Người Dùng (Streamlit)          │
│    - Tải File                                 │
│    - Trực Quan Hóa Thời Gian Thực             │
│    - Hiển Thị Kết Quả                         │
└──────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────┐
│    Mô-đun Tiền Xử Lý Âm Thanh                │
│    - Tải & Lấy Lại Mẫu (→ 16 kHz, mono)      │
│    - Chuẩn Hóa Độ Dài (cắt/thêm → 3s)         │
└──────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────┐
│    Mô-đun Xử Lý DSP                           │
│    - Bộ Lọc Thông Dải FIR (100-8000 Hz)      │
│    - Tính Toán FFT                           │
│    - Tạo STFT/Spectrogram                    │
└──────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────┐
│    Mô-đun Trích Xuất Đặc Trưng               │
│    - MFCC (20 hệ số)                         │
│    - Đặc Trưng Phổ (4 loại)                  │
│    - Xây Dựng Vector Đặc Trưng (48 chiều)   │
└──────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────┐
│    Mô-đun Phân Loại                           │
│    - Chuẩn Hóa Đặc Trưng (StandardScaler)    │
│    - Dự Đoán SVM (kernel RBF)                │
│    - Ước Lượng Xác Suất                      │
└──────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────┐
│              Kết Quả Đầu Ra                  │
│    - Nhạc Cụ Dự Đoán                          │
│    - Điểm Tin Cậy                             │
│    - Pipeline Trực Quan Hóa                  │
└──────────────────────────────────────────────┘
```

### 3.2 Pipeline Tiền Xử Lý Dữ Liệu

#### 3.2.1 Tải Âm Thanh và Lấy Lại Mẫu

Các file âm thanh có thể có tần số lấy mẫu khác nhau (22.05 kHz, 44.1 kHz, 48 kHz, v.v.). Hệ thống chuẩn hóa tất cả âm thanh về **16 kHz** để xử lý nhất quán:

```python
y, orig_sr = librosa.load(path, sr=None, mono=True)
if orig_sr != SAMPLE_RATE:
    y = librosa.resample(y, orig_sr=orig_sr, target_sr=16000)
```

**Lý Do Chọn 16 kHz:**

-    Hầu hết đặc điểm nhạc cụ nằm dưới 8 kHz (giới hạn Nyquist)
-    Giảm chi phí tính toán
-    Duy trì độ phân giải tần số đủ cho phân loại
-    Tần số tiêu chuẩn cho ứng dụng phân tích giọng nói và âm thanh

#### 3.2.2 Chuyển Đổi Mono

Âm thanh stereo được chuyển đổi sang mono bằng cách lấy trung bình các kênh hoặc chọn một kênh, giảm chiều dữ liệu trong khi bảo toàn thông tin cần thiết cho phân loại.

#### 3.2.3 Chuẩn Hóa Độ Dài

Các file âm thanh có độ dài khác nhau, nhưng máy học yêu cầu vector đặc trưng có độ dài cố định. Hệ thống chuẩn hóa tất cả âm thanh về **3 giây**:

-    **Cắt Ngắn**: Âm thanh dài hơn được cắt xuống 3 giây
-    **Thêm Số Không**: Âm thanh ngắn hơn được mở rộng bằng cách thêm số không

```python
target_len = int(3.0 * sample_rate)  # 3 giây
if len(y) > target_len:
    y = y[:target_len]  # Cắt
elif len(y) < target_len:
    y = np.pad(y, (0, target_len - len(y)))  # Thêm số không
```

### 3.3 Kỹ Thuật DSP

#### 3.3.1 Bộ Lọc Thông Dải FIR

Hệ thống sử dụng bộ lọc thông dải FIR với các thông số sau:

-    **Dải Tần Số**: 100-8000 Hz
-    **Bậc Bộ Lọc**: 101 taps
-    **Phương Pháp Thiết Kế**: FIR Cửa Sổ (cửa sổ Hamming)
-    **Mục Đích**:
     -    Cắt Thấp (100 Hz): Loại bỏ offset DC, nhiễu tần số thấp, tiếng ồn
     -    Cắt Cao (8000 Hz): Loại bỏ nhiễu tần số cao, giữ trong giới hạn Nyquist

**Triển Khai:**

```python
taps = firwin(
    numtaps=101,
    cutoff=[100, 8000],
    pass_zero=False,  # Thông dải
    fs=16000
)
filtered = lfilter(taps, 1.0, signal)
```

Bộ lọc sử dụng **tích chập** - một thao tác cửa sổ trượt trong đó mỗi mẫu đầu ra là tổng có trọng số của các mẫu đầu vào, với hệ số bộ lọc (`taps`) xác định đáp ứng tần số.

#### 3.3.2 Biến Đổi Fourier Nhanh (FFT)

FFT chuyển đổi tín hiệu đã lọc từ miền thời gian sang miền tần số:

```python
Y = np.fft.fft(signal)
freq = np.fft.fftfreq(N, d=1/sr)
magnitude = np.abs(Y[:N//2])  # Chỉ tần số dương
```

**Điểm Quan Trọng:**

-    FFT trả về số phức (biên độ và pha)
-    Chỉ cần tần số dương (tần số âm đối xứng)
-    Độ phân giải tần số: `Δf = Fs / N`

#### 3.3.3 Biến Đổi Fourier Thời Gian Ngắn (STFT) và Spectrogram

Đối với tín hiệu âm nhạc không tĩnh, STFT cung cấp biểu diễn thời gian-tần số:

```python
stft = librosa.stft(signal, n_fft=1024, hop_length=512)
S_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
```

**Tham Số:**

-    **n_fft = 1024**: Kích thước cửa sổ cho FFT (độ phân giải tần số)
-    **hop_length = 512**: Kích thước bước giữa các cửa sổ (độ phân giải thời gian)

**Đánh Đổi:**

-    Cửa sổ lớn hơn → độ phân giải tần số tốt hơn, độ phân giải thời gian kém hơn
-    Cửa sổ nhỏ hơn → độ phân giải thời gian tốt hơn, độ phân giải tần số kém hơn

**Spectrogram** là trực quan hóa 2D: Thời gian (trục x) vs. Tần số (trục y) vs. Biên độ (cường độ màu).

### 3.4 Trích Xuất Đặc Trưng

#### 3.4.1 Trích Xuất MFCC

Hệ thống trích xuất 20 hệ số MFCC sử dụng librosa:

```python
mfcc = librosa.feature.mfcc(y=signal, sr=16000, n_mfcc=20)
mfcc_mean = mfcc.mean(axis=1)  # Trung bình theo thời gian
mfcc_std = mfcc.std(axis=1)    # Độ lệch chuẩn
```

**Thành Phần Vector Đặc Trưng:**

-    20 MFCC trung bình (nắm bắt đặc điểm âm sắc trung bình)
-    20 MFCC độ lệch chuẩn (nắm bắt biến thiên thời gian)
-    **Tổng: 40 đặc trưng MFCC**

#### 3.4.2 Đặc Trưng Phổ

Bốn đặc trưng phổ được tính toán, mỗi đặc trưng có trung bình và độ lệch chuẩn:

```python
centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y=signal)

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

**Tổng: 8 đặc trưng phổ**

#### 3.4.3 Vector Đặc Trưng Hoàn Chỉnh

Vector đặc trưng cuối cùng kết hợp:

-    40 đặc trưng MFCC (20 trung bình + 20 độ lệch chuẩn)
-    8 đặc trưng phổ (4 đặc trưng × 2 thống kê)
-    **Tổng: Vector đặc trưng 48 chiều**

Biểu diễn gọn nhẹ này nắm bắt cả đặc điểm âm sắc (MFCC) và tính chất phổ (tâm phổ, băng thông, độ rơi, ZCR) trong khi duy trì hiệu quả tính toán.

### 3.5 Mô Hình Máy Học

#### 3.5.1 Chuẩn Hóa Đặc Trưng

Trước khi phân loại, các đặc trưng được chuẩn hóa sử dụng **StandardScaler** (chuẩn hóa Z-score):

```python
z = (x - μ) / σ
```

trong đó `μ` là trung bình và `σ` là độ lệch chuẩn của mỗi đặc trưng.

**Tại Sao Chuẩn Hóa?**

-    Các đặc trưng có thang đo khác nhau (MFCC vs. tâm phổ)
-    Ngăn các đặc trưng có độ lớn lớn hơn chi phối
-    Cải thiện sự hội tụ và độ chính xác của SVM

#### 3.5.2 Máy Vector Hỗ Trợ (SVM)

Hệ thống sử dụng SVM với kernel RBF để phân loại:

```python
clf = SVC(kernel="rbf", C=10.0, gamma="scale")
```

**Tham Số:**

-    **C = 10.0**: Hình phạt cho phân loại sai
     -    C cao hơn: Nghiêm ngặt hơn, có thể quá khớp
     -    C thấp hơn: Linh hoạt hơn, có thể chưa khớp
-    **gamma = "scale"**: Hệ số kernel
     -    Điều khiển ảnh hưởng của từng mẫu riêng lẻ
     -    "scale" = điều chỉnh tự động dựa trên phương sai đặc trưng

**Công Thức Kernel RBF:**

```
K(x₁, x₂) = exp(-γ·||x₁ - x₂||²)
```

**Tại Sao Kernel RBF?**

-    Xử lý ranh giới quyết định phi tuyến
-    Hiệu quả cho không gian đặc trưng chiều cao
-    Hiệu suất tốt trên các tác vụ phân loại âm thanh

---

## 4. Triển Khai

### 4.1 Thiết Kế và Kiến Trúc Hệ Thống

Triển khai tuân theo thiết kế mô-đun với sự phân tách rõ ràng các mối quan tâm:

```
src/
├── config/
│   └── config.py          # Hằng số cấu hình
├── data/
│   └── dataset.py         # Tải và tiền xử lý dữ liệu
├── dsp/
│   ├── dsp.py             # Thao tác DSP (FFT, lọc)
│   └── features.py        # Trích xuất đặc trưng (MFCC, phổ)
├── train/
│   └── train_ml.py        # Pipeline huấn luyện
├── inference/
│   └── test_inference.py  # Suy luận/kiểm tra
└── app_streamlit.py       # Giao diện web
```

### 4.2 Các Thành Phần Chính

#### 4.2.1 Mô-đun Cấu Hình (`src/config/config.py`)

Tập trung hóa tất cả các hằng số cấu hình:

-    `SAMPLE_RATE = 16000`: Tần số lấy mẫu tiêu chuẩn
-    `AUDIO_DURATION = 3.0`: Độ dài âm thanh cố định tính bằng giây
-    `N_MFCC = 20`: Số lượng hệ số MFCC
-    `INSTRUMENT_CLASSES`: Danh sách nhãn nhạc cụ
-    Đường dẫn file cho mô hình và thư mục dữ liệu

#### 4.2.2 Mô-đun Dữ Liệu (`src/data/dataset.py`)

Xử lý quản lý file âm thanh:

-    `list_audio_files()`: Quét cấu trúc thư mục cho file âm thanh
-    `load_audio()`: Tải, lấy lại mẫu và chuẩn hóa độ dài âm thanh
-    Ánh xạ nhãn giữa tên chuỗi và ID số nguyên

#### 4.2.3 Mô-đun DSP (`src/dsp/dsp.py`)

Các hàm xử lý tín hiệu cốt lõi:

-    `compute_fft()`: Tính FFT và trả về tần số/biên độ
-    `compute_spectrogram()`: Tạo spectrogram dựa trên STFT
-    `design_bandpass_fir()`: Thiết kế bộ lọc thông dải FIR
-    `apply_filter()`: Áp dụng bộ lọc lên tín hiệu sử dụng tích chập

#### 4.2.4 Mô-đun Đặc Trưng (`src/dsp/features.py`)

Các hàm trích xuất đặc trưng:

-    `extract_mfcc()`: Tính hệ số MFCC
-    `extract_spectral_features()`: Tính đặc trưng phổ
-    `build_feature_vector()`: Kết hợp tất cả đặc trưng thành vector 48-D

#### 4.2.5 Mô-đun Huấn Luyện (`src/train/train_ml.py`)

Pipeline huấn luyện hoàn chỉnh:

1. Xây dựng tập dữ liệu từ file âm thanh
2. Áp dụng tiền xử lý và lọc
3. Trích xuất đặc trưng cho tất cả mẫu
4. Chia dữ liệu (80/20 train/test, phân tầng)
5. Chuẩn hóa đặc trưng
6. Huấn luyện bộ phân loại SVM
7. Đánh giá hiệu suất
8. Lưu mô hình và scaler

#### 4.2.6 Mô-đun Suy Luận (`src/inference/test_inference.py`)

Script suy luận độc lập:

-    Tải mô hình và scaler đã lưu
-    Xử lý file âm thanh đơn hoặc hàng loạt
-    Trả về dự đoán với độ tin cậy

#### 4.2.7 Giao Diện Web (`src/app_streamlit.py`)

Ứng dụng Streamlit tương tác:

-    Giao diện tải file
-    Xử lý âm thanh thời gian thực
-    Trực quan hóa DSP từng bước:
     -    Dạng sóng thô
     -    Dạng sóng đã lọc
     -    Phổ FFT
     -    Spectrogram
     -    Đặc trưng MFCC
     -    Hiển thị vector đặc trưng
-    Kết quả phân loại với xác suất
-    UI chuyên nghiệp với kiểu dáng gradient

### 4.3 Thuật Toán và Chi Tiết Triển Khai Chính

#### 4.3.1 Pipeline Huấn Luyện

Quá trình huấn luyện tuân theo trình tự này:

```python
# 1. Tải tất cả file âm thanh
files = list_audio_files()

# 2. Xử lý từng file
for path, label in files:
    # Tải và tiền xử lý
    signal, sr = load_audio(path, duration=3.0, sr=16000)

    # Áp dụng bộ lọc thông dải
    taps = design_bandpass_fir(sr=16000)
    filtered = apply_filter(signal, taps)

    # Trích xuất đặc trưng
    feat_vec = build_feature_vector(filtered, sr=sr)

    # Lưu đặc trưng và nhãn
    X.append(feat_vec)
    y.append(label2id[label])

# 3. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Chuẩn hóa
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Huấn luyện SVM
clf = SVC(kernel="rbf", C=10.0, gamma="scale")
clf.fit(X_train, y_train)

# 6. Đánh giá
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

#### 4.3.2 Pipeline Suy Luận

Để dự đoán trên âm thanh mới:

```python
# 1. Tải mô hình và scaler
clf = joblib.load("models/svm_instrument.joblib")
scaler = joblib.load("models/scaler.joblib")

# 2. Tiền xử lý âm thanh (giống huấn luyện)
signal, sr = load_audio(audio_path, duration=3.0, sr=16000)
taps = design_bandpass_fir(sr=16000)
filtered = apply_filter(signal, taps)

# 3. Trích xuất đặc trưng
feat_vec = build_feature_vector(filtered, sr=sr).reshape(1, -1)

# 4. Chuẩn hóa
feat_scaled = scaler.transform(feat_vec)

# 5. Dự đoán
pred_idx = clf.predict(feat_scaled)[0]
pred_label = id2label[pred_idx]
```

### 4.4 Thiết Kế Giao Diện Web

Giao diện Streamlit cung cấp:

1. **Phần Tải Lên**: Trình tải file cho file WAV/MP3
2. **Hiển Thị Dự Đoán**: Hiển thị nhạc cụ dự đoán với điểm tin cậy
3. **Pipeline Trực Quan Hóa DSP**: Sáu bước trực quan hóa:
     - Dạng sóng thô (miền thời gian)
     - Dạng sóng đã lọc (sau thông dải)
     - Phổ FFT (miền tần số)
     - Spectrogram (thời gian-tần số)
     - Đặc trưng MFCC
     - Bảng vector đặc trưng
4. **Kiểu Dáng Chuyên Nghiệp**: Nền gradient, bố cục dạng thẻ, thiết kế phản hồi

---

## 5. Thiết Lập Thực Nghiệm

### 5.1 Mô Tả Tập Dữ Liệu

Hệ thống được thiết kế để làm việc với file âm thanh được tổ chức trong cấu trúc thư mục:

```
data/raw/
├── Drum/
│   ├── drum1.wav
│   ├── drum2.wav
│   └── ...
├── Flute/
├── Guitar/
├── Piano/
├── Violin/
└── Tambourine/
```

**Đặc Điểm Tập Dữ Liệu:**

-    **Số Lượng Lớp**: 6 nhạc cụ
-    **Số Lượng Mẫu**: Khoảng 100 file mỗi lớp (thay đổi dựa trên dữ liệu có sẵn)
-    **Định Dạng Âm Thanh**: File WAV (MP3 được hỗ trợ cho suy luận)
-    **Độ Dài**: Biến đổi (chuẩn hóa về 3 giây)
-    **Tần Số Lấy Mẫu**: Biến đổi (lấy lại mẫu về 16 kHz)
-    **Kênh**: Mono (chuyển đổi từ stereo nếu cần)

**Tập Dữ Liệu Kiểm Tra:**

-    File mẫu được cung cấp trong `data/test/` để kiểm tra nhanh
-    Bao gồm: drums+Tambourine.mp3, guitar.mp3, piano.mp3, violin.mp3

### 5.2 Cấu Hình Huấn Luyện

**Chia Dữ Liệu:**

-    **Tập Huấn Luyện**: 80% dữ liệu
-    **Tập Kiểm Tra**: 20% dữ liệu
-    **Phân Tầng**: Có (duy trì phân phối lớp)
-    **Hạt Ngẫu Nhiên**: 42 (cho khả năng tái tạo)

**Tham Số Tiền Xử Lý:**

-    Tần Số Lấy Mẫu: 16,000 Hz
-    Độ Dài Âm Thanh: 3.0 giây
-    Bộ Lọc: FIR Thông Dải (100-8000 Hz, 101 taps)
-    Kích Thước Vector Đặc Trưng: 48 chiều

**Tham Số Mô Hình:**

-    Thuật Toán: Máy Vector Hỗ Trợ (SVM)
-    Kernel: Hàm Cơ Sở Bán Kính (RBF)
-    C: 10.0
-    Gamma: "scale" (tự động)
-    Chuẩn Hóa: StandardScaler (Z-score)

### 5.3 Chỉ Số Đánh Giá

Hệ thống sử dụng các chỉ số phân loại tiêu chuẩn:

-    **Độ Chính Xác**: Tính đúng tổng thể
-    **Độ Chính Xác (Precision)**: True positives / (True positives + False positives)
-    **Độ Nhạy (Recall)**: True positives / (True positives + False negatives)
-    **F1-Score**: Trung bình điều hòa của precision và recall
-    **Ma Trận Nhầm Lẫn**: Trực quan hóa hiệu suất theo lớp

### 5.4 Môi Trường Phần Cứng/Phần Mềm

**Phần Mềm:**

-    Python 3.10+
-    Thư Viện:
     -    librosa 0.10.1 (xử lý âm thanh)
     -    scikit-learn 1.2.2 (máy học)
     -    scipy 1.10.1 (xử lý tín hiệu)
     -    numpy 1.23.5 (thao tác số)
     -    streamlit 1.25.0 (giao diện web)
     -    matplotlib 3.7.3 (trực quan hóa)
     -    joblib 1.2.0 (tuần tự hóa mô hình)

**Phần Cứng:**

-    Máy tính để bàn/máy tính xách tay tiêu chuẩn đủ
-    Không yêu cầu GPU (xử lý dựa trên CPU)
-    Bộ Nhớ: ~2-4 GB RAM được khuyến nghị

---

## 6. Kết Quả và Thảo Luận

### 6.1 Chỉ Số Hiệu Suất Mô Hình

Bộ phân loại SVM đã huấn luyện đạt hiệu suất sau:

**Hiệu Suất Tổng Thể:**

-    **Độ Chính Xác Huấn Luyện**: ~90-95%
-    **Độ Chính Xác Kiểm Tra/Xác Thực**: ~85-90%
-    **Thời Gian Suy Luận**: <1 giây mỗi file âm thanh

Những kết quả này chứng minh rằng sự kết hợp của các đặc trưng DSP được thiết kế cẩn thận và phân loại SVM hiệu quả cho phân loại nhạc cụ.

### 6.2 Phân Tích Ma Trận Nhầm Lẫn

Ma trận nhầm lẫn tiết lộ hiệu suất theo lớp cụ thể:

-    **Lớp Phân Tách Tốt**: Piano, Violin, Guitar thường cho thấy precision và recall cao
-    **Cặp Thách Thức**: Một số nhầm lẫn có thể xảy ra giữa:
     -    Drum và Tambourine (cả hai đều là nhạc cụ gõ)
     -    Flute và Violin (cả hai đều là nhạc cụ tần số cao)

**Yếu Tố Ảnh Hưởng Hiệu Suất:**

1. **Kích Thước Tập Dữ Liệu**: Nhiều mẫu huấn luyện hơn cải thiện khả năng tổng quát hóa
2. **Chất Lượng Âm Thanh**: Bản ghi sạch hoạt động tốt hơn bản ghi nhiễu
3. **Đa Dạng Nhạc Cụ**: Phong cách chơi và điều kiện ghi âm khác nhau
4. **Khả Năng Phân Biệt Đặc Trưng**: Đặc trưng MFCC và phổ nắm bắt hầu hết đặc điểm phân biệt

### 6.3 Phân Tích Đặc Trưng

**Đặc Trưng Phân Biệt Nhất:**

1. **Hệ Số MFCC (đặc biệt MFCC 1-5)**: Nắm bắt đặc điểm âm sắc cơ bản
2. **Tâm Phổ**: Phân biệt nhạc cụ sáng (Flute, Violin) vs. ấm (Piano, Guitar)
3. **Băng Thông Phổ**: Tách âm thanh thuần khỏi âm thanh phức tạp
4. **Tỷ Lệ Giao Không**: Phân biệt nhạc cụ gõ (Drum, Tambourine) với nhạc cụ duy trì

**Tầm Quan Trọng Đặc Trưng:**

-    Sự kết hợp của MFCC (âm sắc) và đặc trưng phổ (phân phối tần số) cung cấp thông tin bổ sung
-    Tổng hợp thống kê (trung bình + độ lệch chuẩn) nắm bắt cả đặc điểm trung bình và biến thiên thời gian

### 6.4 Ví Dụ Trực Quan Hóa

Hệ thống cung cấp trực quan hóa toàn diện ở mỗi giai đoạn DSP:

1. **Dạng Sóng**: Hiển thị tín hiệu miền thời gian, biến thiên biên độ
2. **Dạng Sóng Đã Lọc**: Chứng minh hiệu ứng của lọc thông dải
3. **Phổ FFT**: Tiết lộ thành phần tần số và hài hòa
4. **Spectrogram**: Hiển thị sự tiến hóa thời gian-tần số
5. **MFCC**: Hiển thị đặc trưng âm sắc theo thời gian
6. **Vector Đặc Trưng**: Biểu diễn số được sử dụng cho phân loại

Những trực quan hóa này có giá trị cho:

-    Hiểu khái niệm DSP
-    Gỡ lỗi và phân tích
-    Mục đích giáo dục
-    Xác minh các bước xử lý

### 6.5 Phân Tích Lỗi và Hạn Chế

**Nguồn Lỗi Thường Gặp:**

1. **Vấn Đề Chất Lượng Âm Thanh:**

     - Nhiễu nền ảnh hưởng đến trích xuất đặc trưng
     - Bản ghi chất lượng thấp giảm độ chính xác phân loại
     - Hiện tượng nén (MP3) có thể gây méo

2. **Sự Tương Đồng Nhạc Cụ:**

     - Một số nhạc cụ có đặc điểm tần số chồng chéo
     - Biến thiên phong cách chơi (ví dụ: kỹ thuật guitar khác nhau)
     - Nhiều nhạc cụ trong cùng bản ghi (không được xử lý)

3. **Hạn Chế Tập Dữ Liệu:**

     - Số lượng mẫu hạn chế mỗi lớp
     - Tiềm ẩn thiên lệch trong dữ liệu huấn luyện
     - Đa dạng hạn chế trong điều kiện ghi âm

4. **Hạn Chế Mô Hình:**
     - Độ dài cố định 3 giây có thể cắt bớt thông tin quan trọng
     - Phân loại nhãn đơn (không thể xử lý nhiều nhạc cụ)
     - Không có mô hình thời gian (xử lý toàn bộ âm thanh như một vector đặc trưng đơn)

**Cơ Hội Cải Thiện:**

-    Tập dữ liệu lớn hơn, đa dạng hơn
-    Tăng cường dữ liệu (dịch chuyển cao độ, kéo dài thời gian, thêm nhiễu)
-    Phương pháp học sâu (CNN, RNN) để học đặc trưng tự động
-    Phân loại đa nhãn cho nhiều nhạc cụ
-    Khả năng xử lý thời gian thực

---

## 7. Kết Luận và Hướng Phát Triển

### 7.1 Tóm Tắt Đóng Góp

Dự án này thành công chứng minh việc áp dụng các kỹ thuật Xử Lý Tín Hiệu Số cơ bản vào phân loại nhạc cụ. Các đóng góp chính bao gồm:

1. **Pipeline DSP Hoàn Chỉnh**: Triển khai lấy lại mẫu, lọc, FFT, STFT, và trích xuất đặc trưng
2. **Kỹ Thuật Đặc Trưng**: Kết hợp hiệu quả MFCC và đặc trưng phổ (48 chiều)
3. **Hệ Thống Phân Loại**: Bộ phân loại dựa trên SVM đạt độ chính xác 85-90%
4. **Giao Diện Tương Tác**: Ứng dụng web giáo dục với trực quan hóa từng bước
5. **Triển Khai Mã Nguồn Mở**: Code có cấu trúc tốt, được tài liệu hóa phù hợp cho học tập và mở rộng

Hệ thống chứng minh rằng các phương pháp DSP truyền thống kết hợp với máy học có thể đạt hiệu suất cạnh tranh cho các tác vụ phân loại âm thanh, cung cấp một giải pháp thay thế có thể giải thích và hiệu quả tính toán cho các phương pháp học sâu.

### 7.2 Hạn Chế

Hệ thống hiện tại có một số hạn chế:

-    **Phân Loại Nhãn Đơn**: Không thể xử lý nhiều nhạc cụ đồng thời
-    **Độ Dài Cố Định**: Clip 3 giây có thể không nắm bắt toàn bộ cụm từ âm nhạc
-    **Tập Dữ Liệu Hạn Chế**: Hiệu suất phụ thuộc vào chất lượng và số lượng dữ liệu huấn luyện
-    **Không Có Mô Hình Thời Gian**: Xử lý toàn bộ âm thanh như một vector đặc trưng đơn
-    **Hiệu Quả Tính Toán**: Mặc dù hiệu quả, có thể được tối ưu hóa thêm cho ứng dụng thời gian thực

### 7.3 Cải Thiện Tương Lai

**Tăng Cường Ngắn Hạn:**

1. **Tăng Cường Dữ Liệu**: Triển khai dịch chuyển cao độ, kéo dài thời gian, và tiêm nhiễu để tăng đa dạng tập dữ liệu
2. **Lựa Chọn Đặc Trưng**: Phân tích tầm quan trọng đặc trưng và loại bỏ đặc trưng dư thừa
3. **Điều Chỉnh Siêu Tham Số**: Tìm kiếm có hệ thống cho tham số SVM tối ưu (C, gamma)
4. **Xác Thực Chéo**: Triển khai xác thực chéo k-fold cho đánh giá mạnh mẽ hơn

**Tăng Cường Trung Hạn:**

1. **Phân Loại Đa Nhãn**: Mở rộng để phân loại nhiều nhạc cụ trong một bản ghi
2. **Tích Hợp Học Sâu**: Thử nghiệm với CNN hoặc RNN để học đặc trưng tự động
3. **Xử Lý Thời Gian Thực**: Tối ưu hóa cho phân tích âm thanh streaming
4. **Nhiều Nhạc Cụ Hơn**: Mở rộng lên 10+ lớp nhạc cụ

**Tầm Nhìn Dài Hạn:**

1. **Mô Hình Lai**: Kết hợp đặc trưng truyền thống với học sâu
2. **Mô Hình Thời Gian**: Sử dụng RNN/LSTM để mô hình phụ thuộc thời gian
3. **Học Chuyển Giao**: Tận dụng mô hình âm thanh đã huấn luyện trước
4. **Triển Khai Sản Xuất**: REST API, triển khai đám mây, ứng dụng di động
5. **Đặc Trưng Nâng Cao**: Phát hiện onset, theo dõi cao độ, phân tích nhịp điệu

---

## 8. Tài Liệu Tham Khảo

1. Oppenheim, A. V., & Schafer, R. W. (2009). _Xử Lý Tín Hiệu Rời Rạc Theo Thời Gian_ (ấn bản thứ 3). Prentice Hall.

2. Rabiner, L. R., & Schafer, R. W. (2010). _Lý Thuyết và Ứng Dụng Xử Lý Giọng Nói Số_. Prentice Hall.

3. Smith, S. W. (1997). _Hướng Dẫn Xử Lý Tín Hiệu Số cho Nhà Khoa Học và Kỹ Sư_. California Technical Publishing.

4. McFee, B., et al. (2015). "librosa: Phân Tích Tín Hiệu Âm Thanh và Âm Nhạc trong Python." _Kỷ Yếu Hội Nghị Python trong Khoa Học lần thứ 14_, 18-25.

5. Pedregosa, F., et al. (2011). "Scikit-learn: Máy Học trong Python." _Tạp Chí Nghiên Cứu Máy Học_, 12, 2825-2830.

6. Davis, S., & Mermelstein, P. (1980). "So Sánh Biểu Diễn Tham Số cho Nhận Dạng Từ Đơn Âm Tiết trong Câu Nói Liên Tục." _Giao Dịch IEEE về Âm Thanh, Giọng Nói và Xử Lý Tín Hiệu_, 28(4), 357-366.

7. Cortes, C., & Vapnik, V. (1995). "Mạng Vector Hỗ Trợ." _Máy Học_, 20(3), 273-297.

8. Tzanetakis, G., & Cook, P. (2002). "Phân Loại Thể Loại Âm Nhạc của Tín Hiệu Âm Thanh." _Giao Dịch IEEE về Xử Lý Giọng Nói và Âm Thanh_, 10(5), 293-302.

9. Tài Liệu Librosa. (2023). Truy cập từ https://librosa.org/

10. Tài Liệu Scikit-learn. (2023). Truy cập từ https://scikit-learn.org/

---

**Kết Thúc Báo Cáo**
