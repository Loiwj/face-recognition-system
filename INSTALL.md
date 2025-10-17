# Hướng dẫn Cài đặt Hệ thống Nhận diện Khuôn mặt

## Yêu cầu Hệ thống

### Phần cứng
- **CPU**: Intel i5 hoặc AMD Ryzen 5 trở lên
- **RAM**: Tối thiểu 8GB, khuyến nghị 16GB
- **GPU**: NVIDIA GPU với CUDA (tùy chọn, để tăng tốc độ)
- **Ổ cứng**: Tối thiểu 5GB trống

### Phần mềm
- **Python**: 3.8 trở lên
- **Operating System**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+

## Cài đặt

### 1. Clone Repository
```bash
git clone https://github.com/your-username/face-recognition-system.git
cd face-recognition-system
```

### 2. Tạo Môi trường Ảo

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Cài đặt Dependencies
```bash
pip install -r requirements.txt
```

### 4. Khởi tạo FaceNet Model
```bash
# FaceNet model sẽ được tự động tải khi chạy ứng dụng lần đầu
# keras-facenet sẽ tự động download và cache model
# Không cần tải model thủ công
```

### 5. Kiểm tra Cài đặt
```bash
python -c "
import sys
sys.path.append('src')
from modules.face_detection import FaceDetectorFactory
from modules.facenet_embedding import FaceNetFactory
print('All modules imported successfully!')
"
```

## Chạy Ứng dụng

### 1. Khởi động Web Application
```bash
python run.py
```

### 2. Truy cập Ứng dụng
Mở trình duyệt và truy cập: http://localhost:5000

### 3. Các tùy chọn khởi động
```bash
# Chạy trên port khác
python run.py --port 8080

# Chạy trên host khác
python run.py --host 0.0.0.0 --port 5000

# Chạy ở chế độ debug
python run.py --debug

# Chỉ khởi tạo models (để test)
python run.py --init-models
```

## Cấu hình

### 1. File cấu hình
Chỉnh sửa file `config.py` để thay đổi cấu hình:

```python
# Thay đổi detector
face_detection.detector_type = "mtcnn"  # mediapipe, mtcnn, retinaface, opencv

# Thay đổi threshold
comparison.threshold = 0.7

# Thay đổi port
web.port = 8080
```

### 2. Biến môi trường
```bash
# Development
export FLASK_ENV=development

# Production
export FLASK_ENV=production
```

## Training Model

### 1. Chuẩn bị Dataset
```
data/training/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person2/
│   ├── image1.jpg
│   └── ...
└── ...
```

### 2. Training từ đầu
```bash
python src/training/train_facenet.py \
    --dataset_path data/training \
    --output_dir models/trained \
    --epochs 100 \
    --batch_size 32
```

### 3. Fine-tuning
```bash
python src/training/fine_tune_facenet.py \
    --pretrained_model models/facenet_keras.h5 \
    --custom_data data/custom \
    --output_dir models/fine_tuned \
    --fine_tune_epochs 20
```

### 4. Đánh giá Model
```bash
python src/training/evaluate_model.py \
    --model_path models/trained/custom_facenet.h5 \
    --test_data data/test \
    --output_dir evaluation_results
```

## Sử dụng API

### 1. Đăng ký người dùng
```bash
curl -X POST http://localhost:5000/api/enroll \
    -H "Content-Type: application/json" \
    -d '{
        "name": "John Doe",
        "image": "base64_encoded_image"
    }'
```

### 2. Nhận diện
```bash
curl -X POST http://localhost:5000/api/recognize \
    -H "Content-Type: application/json" \
    -d '{
        "image": "base64_encoded_image"
    }'
```

### 3. Quản lý Database
```bash
# Lấy danh sách người dùng
curl http://localhost:5000/api/database/persons

# Xóa người dùng
curl -X DELETE http://localhost:5000/api/database/persons/John%20Doe

# Backup database
curl -X POST http://localhost:5000/api/database/backup
```

## Xử lý Sự cố

### 1. Lỗi Import Module
```bash
# Kiểm tra Python path
python -c "import sys; print(sys.path)"

# Thêm src vào path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### 2. Lỗi Camera
- Kiểm tra quyền truy cập camera
- Đảm bảo không có ứng dụng khác đang sử dụng camera
- Thử với trình duyệt khác

### 3. Lỗi Model Loading
```bash
# Kiểm tra file model
ls -la models/

# Tải lại model
rm models/facenet_keras.h5
# Tải lại theo hướng dẫn trên
```

### 4. Lỗi Memory
- Giảm batch_size trong config
- Sử dụng detector nhẹ hơn (mediapipe thay vì mtcnn)
- Đóng các ứng dụng khác

### 5. Lỗi Database
```bash
# Xóa database cũ
rm face_database.db

# Khởi động lại ứng dụng
python run.py
```

## Performance Tuning

### 1. Tăng tốc độ
- Sử dụng GPU nếu có
- Giảm kích thước ảnh input
- Sử dụng detector nhanh (mediapipe)

### 2. Tăng độ chính xác
- Sử dụng detector chính xác (retinaface)
- Tăng threshold
- Sử dụng model lớn hơn

### 3. Tối ưu Memory
- Giảm batch_size
- Sử dụng model nhẹ (mobilenet)
- Xóa cache định kỳ

## Deployment

### 1. Production
```bash
# Cài đặt gunicorn
pip install gunicorn

# Chạy với gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 2. Docker
```bash
# Build image
docker build -t face-recognition .

# Chạy container
docker run -p 5000:5000 face-recognition
```

### 3. Nginx
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Hỗ trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra logs trong thư mục `logs/`
2. Tạo issue trên GitHub
3. Liên hệ qua email: support@example.com

## License

MIT License - Xem file LICENSE để biết thêm chi tiết.
