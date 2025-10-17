# Hệ thống Nhận diện Khuôn mặt với FaceNet

Dự án xây dựng hệ thống nhận diện khuôn mặt hoàn chỉnh sử dụng FaceNet, có khả năng học (đăng ký) khuôn mặt người dùng mới từ ảnh thông thường và nhận diện trong thời gian thực.

## ✨ Tính năng chính

- 🎯 **Đăng ký Người dùng**: Chấp nhận ảnh thông thường, tự động phát hiện và trích xuất khuôn mặt
- 📹 **Nhận diện Thời gian thực**: Nhận diện nhiều người trong luồng video với tốc độ cao
- 🔍 **Xác định Người lạ**: Phân biệt người đã đăng ký và người lạ
- 🏗️ **Kiến trúc Module hóa**: Dễ dàng thay thế và nâng cấp từng thành phần
- 🎓 **Training và Fine-tuning**: Hỗ trợ training FaceNet từ đầu hoặc fine-tune

## 🚀 Cài đặt nhanh

### 1. Clone repository
```bash
git clone https://github.com/your-username/face-recognition-system.git
cd face-recognition-system
```

### 2. Tạo môi trường ảo
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Khởi tạo FaceNet Model
```bash
# FaceNet model sẽ được tự động tải khi chạy ứng dụng lần đầu
# Không cần tải model thủ công, keras-facenet sẽ tự động xử lý
```

### 5. Chạy ứng dụng
```bash
python app.py
```

Truy cập: http://localhost:5000

## 📖 Cách sử dụng

### Đăng ký người dùng mới
1. Truy cập trang **Đăng ký**
2. Nhập tên định danh
3. Upload ảnh thông thường (có thể chứa nhiều người)
4. Hệ thống tự động phát hiện và cho phép chọn khuôn mặt cần đăng ký

### Nhận diện
1. Truy cập trang **Nhận diện**
2. Hệ thống kích hoạt webcam
3. Đưa khuôn mặt vào khung hình
4. Xem kết quả nhận diện với bounding box và tên

## 🏗️ Kiến trúc

```
Ảnh đầu vào ➡️ [Face Detection] ➡️ [Preprocessing] ➡️ [FaceNet Embedding] ➡️ [Comparison & Classification]
```

### Các Module chính:
- **Module 1**: Face Detection (Mediapipe, MTCNN, RetinaFace, YOLOv8)
- **Module 2**: Image Preprocessing (Crop, Resize, Normalize, Augmentation)
- **Module 3**: FaceNet Embedding (Inception ResNet, MobileNet)
- **Module 4**: Comparison & Classification (Cosine, Euclidean, SVM, KNN)

## 🎓 Training

### Training FaceNet từ đầu
```bash
python train_facenet.py --dataset_path ./data/training --epochs 100 --batch_size 32
```

### Fine-tuning pre-trained model
```bash
python fine_tune_facenet.py --pretrained_model models/facenet_keras.h5 --custom_data ./data/custom
```

## 📁 Cấu trúc dự án

```
face-recognition-system/
├── app.py                 # Flask web application
├── requirements.txt       # Dependencies
├── README.md             # Documentation
├── models/               # FaceNet models
├── data/                 # Training data
├── src/                  # Source code
│   ├── modules/          # Core modules
│   │   ├── face_detection.py
│   │   ├── preprocessing.py
│   │   ├── facenet_embedding.py
│   │   └── comparison.py
│   ├── database/         # Database modules
│   ├── training/         # Training scripts
│   └── utils/            # Utilities
├── static/               # Web assets
├── templates/            # HTML templates
└── tests/                # Test files
```

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## 📄 License

MIT License
