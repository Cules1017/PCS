# Phát hiện Truy cập Bất thường trong Mạng WiFi

Ứng dụng web demo cho phép phân tích và phát hiện truy cập bất thường trong mạng WiFi sử dụng AI.

## Tính năng

- Upload và phân tích file dump WiFi
- Lọc và hiển thị danh sách Access Points theo nhiều tiêu chí
- Đánh giá rủi ro sử dụng mô hình AI đã được huấn luyện
- Trực quan hóa kết quả bằng biểu đồ

## Cài đặt

1. Clone repository:
```bash
git clone [URL repository]
cd [tên thư mục]
```

2. Tạo môi trường ảo Python:
```bash
python -m venv venv
```

3. Kích hoạt môi trường ảo:
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

4. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Huấn luyện và Lưu Mô hình

1. Train mô hình và lưu:
```bash
python save_model.py
```

Quá trình này sẽ:
- Tải dataset NSL-KDD
- Huấn luyện mô hình Random Forest
- Lưu mô hình và các transformer vào thư mục `models/`

## Chạy Ứng dụng

1. Khởi động server Flask:
```bash
python app.py
```

2. Mở trình duyệt và truy cập:
```
http://localhost:5000
```

## Sử dụng

1. **Trang Upload**:
   - Kéo thả hoặc chọn file dump WiFi
   - Hỗ trợ định dạng .csv và .dump

2. **Trang Phân tích**:
   - Xem danh sách Access Points
   - Lọc theo SSID, MAC Address, loại mã hóa
   - Sắp xếp theo các tiêu chí khác nhau

3. **Trang Kết quả**:
   - Xem biểu đồ phân tích rủi ro
   - Danh sách chi tiết đánh giá từng AP
   - Mức độ rủi ro và xác suất

## Cấu trúc File CSV

File CSV đầu vào cần có các cột sau:
- SSID: Tên của Access Point
- MAC: Địa chỉ MAC
- Encryption: Loại mã hóa (WPA2, WPA, WEP, Open)
- Signal_Strength: Cường độ tín hiệu (dBm)
- Channel: Kênh WiFi

## Đóng góp

Vui lòng gửi pull request cho bất kỳ cải tiến nào.

## Giấy phép

MIT License

## Tác giả

[Tên tác giả]

## Hướng dẫn chi tiết

### Bước 1: Cài đặt môi trường
```bash
# Tạo môi trường ảo Python
python -m venv venv

# Kích hoạt môi trường ảo (Windows)
venv\Scripts\activate

# Kích hoạt môi trường ảo (Linux/Mac)
# source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Bước 2: Train và lưu mô hình
```bash
# Huấn luyện mô hình Random Forest và lưu vào thư mục models/
python save_model.py
```

### Bước 3: Chạy ứng dụng web
```bash
# Khởi động Flask server
python app.py

# Truy cập ứng dụng tại http://localhost:5000