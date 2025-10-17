# Hệ thống Nhận diện Khuôn mặt với FaceNet

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Hệ thống nhận diện khuôn mặt hoàn chỉnh sử dụng FaceNet với PyTorch, có khả năng học (đăng ký) khuôn mặt người dùng mới từ ảnh thông thường và nhận diện trong thời gian thực.

## ✨ Tính năng chính

- 🎯 **Đăng ký Người dùng**: Chấp nhận ảnh thông thường, tự động phát hiện và trích xuất khuôn mặt
- 📹 **Nhận diện Thời gian thực**: Nhận diện và hiển thị tên của nhiều người trong luồng video
- 🔍 **Xác định Người lạ**: Phân biệt người đã đăng ký và người lạ
- 🏗️ **Kiến trúc Module hóa**: Dễ dàng thay thế hoặc nâng cấp từng thành phần
- 🎓 **Training và Fine-tuning**: Hỗ trợ training FaceNet từ đầu hoặc fine-tune từ pre-trained model
- ⚙️ **Cài đặt Linh hoạt**: Điều chỉnh threshold, phương pháp so sánh trực tiếp trên giao diện

## 🏗️ Kiến trúc Hệ thống

### Pipeline Xử lý Cốt lõi
```
Ảnh đầu vào ➡️ [Phát hiện khuôn mặt] ➡️ [Tiền xử lý] ➡️ [FaceNet Embedding] ➡️ [So sánh & Phân loại]
```

### Các Module chính

1. **Module 1: Phát hiện khuôn mặt**
   - Mediapipe (khuyến nghị cho Real-time)
   - MTCNN (cân bằng tốc độ và độ chính xác)
   - OpenCV DNN (fallback)

2. **Module 2: Tiền xử lý ảnh**
   - Crop và Alignment
   - Resize về 160x160 pixels
   - Normalization và Data Augmentation

3. **Module 3: FaceNet Embedding**
   - Sử dụng facenet-pytorch
   - Pre-trained models: VGGFace2, CASIA-WebFace
   - Embedding vector 512 chiều

4. **Module 4: So sánh và Phân loại**
   - Cosine Similarity
   - Euclidean Distance
   - Manhattan Distance
   - Threshold tuning cho Unknown detection

## 💻 Ngăn xếp Công nghệ

### Backend
- **Python 3.10+**
- **PyTorch 2.2+** + facenet-pytorch
- **Flask** web framework
- **SQLite** database
- **OpenCV** + Mediapipe

### Frontend
- **HTML5, CSS3, JavaScript**
- **Bootstrap 5**
- **Real-time** WebSocket
- **Responsive Design**

## 🚀 Cài đặt và Chạy thử

### 1. Clone repository
```bash
git clone https://github.com/Loiwj/face-recognition-system.git
cd face-recognition-system
```

### 2. Tạo virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Chạy ứng dụng
```bash
python run.py
```

Truy cập: http://localhost:5000

## 📖 Cách sử dụng

### Đăng ký Người dùng
1. Vào trang **Enrollment**
2. Upload ảnh hoặc chụp trực tiếp từ webcam
3. Nhập tên người dùng
4. Hệ thống tự động phát hiện và lưu embedding

### Nhận diện Thời gian thực
1. Vào trang **Recognition**
2. Điều chỉnh tham số:
   - **Phương pháp**: Cosine Similarity, Euclidean Distance, Manhattan Distance
   - **Ngưỡng Tin cậy**: 0.1 - 1.0
   - **Ngưỡng Người lạ**: 0.1 - 1.0
3. Nhấn "Bắt đầu" để bắt đầu nhận diện
4. Hệ thống hiển thị tên và confidence score

### Quản lý Database
- Xem danh sách người đã đăng ký
- Xóa người dùng
- Backup và Export database

## 🔧 Cấu hình

### Tham số quan trọng
- **Threshold**: Ngưỡng tin cậy để chấp nhận nhận diện (mặc định: 0.6)
- **Unknown Threshold**: Ngưỡng để phân loại người lạ (mặc định: 0.85)
- **Face Detection**: Mediapipe (nhanh) hoặc MTCNN (chính xác)

### Tùy chỉnh Model
```python
# Thay đổi FaceNet model
face_embedder = FaceNetFactory.create_embedder("vggface2")  # hoặc "casia-webface"

# Thay đổi phương pháp so sánh
face_comparator = ComparisonFactory.create_comparator("cosine_similarity")
```

## 📊 Hiệu suất

- **Tốc độ**: ~50-100ms per frame (CPU)
- **Độ chính xác**: >95% với điều kiện ánh sáng tốt
- **Memory**: ~500MB RAM
- **Storage**: ~100MB cho model + database

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Liên hệ

- **GitHub**: [@Loiwj](https://github.com/Loiwj)
- **Repository**: [face-recognition-system](https://github.com/Loiwj/face-recognition-system)

## 🙏 Acknowledgments

- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) - FaceNet implementation
- [Mediapipe](https://mediapipe.dev/) - Face detection
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Flask](https://flask.palletsprojects.com/) - Web framework

---

⭐ **Nếu dự án hữu ích, hãy cho một star!** ⭐