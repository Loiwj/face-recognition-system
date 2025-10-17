"""
Flask Web Application cho Hệ thống Nhận diện Khuôn mặt
"""

import os
import sys
import cv2
import numpy as np
import base64
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
from modules.face_detection import FaceDetectorFactory
from modules.preprocessing import FacePreprocessor, create_preprocessing_config, PreprocessingMode
from modules.facenet_pytorch_embedding import FaceNetFactory
from modules.comparison import ComparisonFactory
from database.face_database import FaceDatabase, PersonRecord

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
CORS(app)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for models
face_detector = None
face_preprocessor = None
face_embedder = None
face_comparator = None
face_database = None

# Recognition settings
recognition_settings = {
    "method": "cosine_similarity",
    "threshold": 0.6,
    "unknown_threshold": 0.85
}

# Initialize models
def update_comparator():
    """Cập nhật comparator với settings mới"""
    global face_comparator
    try:
        face_comparator = ComparisonFactory.create_comparator(
            recognition_settings["method"], 
            threshold=recognition_settings["threshold"], 
            unknown_threshold=recognition_settings["unknown_threshold"]
        )
        logger.info(f"Updated comparator with settings: {recognition_settings}")
        return True
    except Exception as e:
        logger.error(f"Error updating comparator: {e}")
        return False

def initialize_models():
    """Khởi tạo các models"""
    global face_detector, face_preprocessor, face_embedder, face_comparator, face_database
    
    try:
        # Initialize face detector
        face_detector = FaceDetectorFactory.create_detector('mediapipe')
        logger.info(f"Initialized face detector: {face_detector.get_name()}")
        
        # Initialize preprocessor
        config = create_preprocessing_config(PreprocessingMode.INFERENCE)
        face_preprocessor = FacePreprocessor(config)
        logger.info("Initialized face preprocessor")
        
        # Initialize FaceNet embedder
        face_embedder = FaceNetFactory.create_embedder("vggface2")
        logger.info("Initialized FaceNet embedder")
        
        # Initialize comparator with current settings
        face_comparator = ComparisonFactory.create_comparator(
            recognition_settings["method"], 
            threshold=recognition_settings["threshold"], 
            unknown_threshold=recognition_settings["unknown_threshold"]
        )
        logger.info(f"Initialized face comparator with settings: {recognition_settings}")
        
        # Initialize database
        face_database = FaceDatabase("data/face_database.db")
        logger.info("Initialized face database")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        return False

# Utility functions
def allowed_file(filename):
    """Kiểm tra file có được phép upload không"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def base64_to_image(base64_string):
    """Chuyển đổi base64 string thành OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        logger.error(f"Error converting base64 to image: {e}")
        return None

def image_to_base64(image):
    """Chuyển đổi OpenCV image thành base64 string"""
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

def process_image_for_enrollment(image, person_name):
    """Xử lý ảnh cho enrollment"""
    try:
        # Detect faces
        faces = face_detector.detect_faces(image)
        
        if not faces:
            return {"success": False, "message": "Không phát hiện được khuôn mặt trong ảnh"}
        
        # Process faces
        processed_faces = face_preprocessor.preprocess_faces(image, faces, PreprocessingMode.ENROLLMENT)
        
        if not processed_faces:
            return {"success": False, "message": "Không thể xử lý khuôn mặt"}
        
        # Extract embeddings
        embeddings = []
        for processed_face in processed_faces:
            embedding = face_embedder.extract_embedding(processed_face)
            if embedding is not None:
                embeddings.append(embedding)
        
        if not embeddings:
            return {"success": False, "message": "Không thể trích xuất embedding"}
        
        # Add to database
        person_id = face_database.add_person(
            name=person_name,
            embeddings=embeddings,
            image_paths=[],  # No file paths for web uploads
            confidence_scores=[face.confidence for face in faces[:len(embeddings)]],
            metadata={
                "enrollment_method": "web_upload",
                "enrollment_date": datetime.now().isoformat(),
                "faces_detected": len(faces),
                "embeddings_created": len(embeddings)
            }
        )
        
        return {
            "success": True,
            "message": f"Đã đăng ký thành công {person_name}",
            "person_id": person_id,
            "faces_detected": len(faces),
            "embeddings_created": len(embeddings)
        }
        
    except Exception as e:
        logger.error(f"Error in enrollment: {e}")
        return {"success": False, "message": f"Lỗi xử lý: {str(e)}"}

def process_image_for_recognition(image):
    """Xử lý ảnh cho recognition"""
    try:
        logger.info(f"Processing image for recognition, shape: {image.shape}")
        
        # Detect faces
        faces = face_detector.detect_faces(image)
        logger.info(f"Detected {len(faces)} faces")
        
        if not faces:
            return {"success": False, "message": "Không phát hiện được khuôn mặt"}
        
        # Process faces
        processed_faces = face_preprocessor.preprocess_faces(image, faces, PreprocessingMode.INFERENCE)
        logger.info(f"Processed {len(processed_faces)} faces")
        
        if not processed_faces:
            return {"success": False, "message": "Không thể xử lý khuôn mặt"}
        
        # Get database embeddings
        database_embeddings = face_database.get_all_embeddings()
        logger.info(f"Database has {len(database_embeddings)} identities")
        
        if not database_embeddings:
            return {"success": False, "message": "Database trống, chưa có người nào được đăng ký"}
        
        # Recognize faces
        results = []
        for i, processed_face in enumerate(processed_faces):
            logger.info(f"Processing face {i+1}/{len(processed_faces)}")
            embedding = face_embedder.extract_embedding(processed_face)
            if embedding is not None:
                logger.info(f"Extracted embedding for face {i+1}, shape: {embedding.shape}")
                result = face_comparator.compare_face(embedding, database_embeddings)
                logger.info(f"Recognition result: {result.identity}, confidence: {result.confidence}, is_unknown: {result.is_unknown}")
                logger.info(f"Comparator unknown_threshold: {face_comparator.config.unknown_threshold}")
                
                # Extract bbox coordinates from tuple
                bbox = faces[i].bbox
                logger.info(f"Face {i+1} bbox type: {type(bbox)}, value: {bbox}")
                if isinstance(bbox, tuple) and len(bbox) == 4:
                    x, y, width, height = bbox
                else:
                    # Fallback if bbox is not tuple
                    x, y, width, height = 0, 0, 100, 100
                
                results.append({
                    "face_index": i,
                    "bbox": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(width),
                        "height": int(height)
                    },
                    "confidence": float(faces[i].confidence),
                    "recognition_result": result.to_dict()
                })
        
        return {
            "success": True,
            "faces_detected": len(faces),
            "recognition_results": results
        }
        
    except Exception as e:
        logger.error(f"Error in recognition: {e}")
        return {"success": False, "message": f"Lỗi xử lý: {str(e)}"}

# Routes
@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/enrollment')
def enrollment():
    """Trang đăng ký"""
    return render_template('enrollment.html')

@app.route('/recognition')
def recognition():
    """Trang nhận diện"""
    return render_template('recognition.html')

@app.route('/database')
def database_view():
    """Trang quản lý database"""
    try:
        persons = face_database.get_all_persons()
        stats = face_database.get_database_stats()
        return render_template('database.html', persons=persons, stats=stats)
    except Exception as e:
        logger.error(f"Error loading database view: {e}")
        flash(f"Lỗi tải database: {str(e)}", 'error')
        return render_template('database.html', persons=[], stats={})

@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    """API endpoint cho enrollment"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data or 'name' not in data:
            return jsonify({"success": False, "message": "Thiếu dữ liệu"})
        
        # Convert base64 to image
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({"success": False, "message": "Lỗi chuyển đổi ảnh"})
        
        # Process enrollment
        result = process_image_for_enrollment(image, data['name'])
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in enrollment API: {e}")
        return jsonify({"success": False, "message": f"Lỗi server: {str(e)}"})

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """API endpoint cho recognition"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"success": False, "message": "Thiếu dữ liệu"})
        
        # Convert base64 to image
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({"success": False, "message": "Lỗi chuyển đổi ảnh"})
        
        # Process recognition
        result = process_image_for_recognition(image)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in recognition API: {e}")
        return jsonify({"success": False, "message": f"Lỗi server: {str(e)}"})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """API endpoint cho file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "message": "Không có file"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "message": "Không có file được chọn"})
        
        if file and allowed_file(file.filename):
            # Read image
            image_data = file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({"success": False, "message": "Lỗi đọc ảnh"})
            
            # Get person name
            person_name = request.form.get('name', 'Unknown')
            
            # Process enrollment
            result = process_image_for_enrollment(image, person_name)
            
            return jsonify(result)
        else:
            return jsonify({"success": False, "message": "Định dạng file không được hỗ trợ"})
            
    except Exception as e:
        logger.error(f"Error in upload API: {e}")
        return jsonify({"success": False, "message": f"Lỗi server: {str(e)}"})

@app.route('/api/database/persons', methods=['GET'])
def api_get_persons():
    """API lấy danh sách người"""
    try:
        persons = face_database.get_all_persons()
        return jsonify({
            "success": True,
            "persons": [person.to_dict() for person in persons]
        })
    except Exception as e:
        logger.error(f"Error getting persons: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/database/persons/<name>', methods=['DELETE'])
def api_delete_person(name):
    """API xóa người"""
    try:
        success = face_database.delete_person(name)
        if success:
            return jsonify({"success": True, "message": f"Đã xóa {name}"})
        else:
            return jsonify({"success": False, "message": f"Không tìm thấy {name}"})
    except Exception as e:
        logger.error(f"Error deleting person: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/database/stats', methods=['GET'])
def api_get_stats():
    """API lấy thống kê database"""
    try:
        stats = face_database.get_database_stats()
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/database/backup', methods=['POST'])
def api_backup_database():
    """API backup database"""
    try:
        backup_path = face_database.backup_database()
        return jsonify({"success": True, "message": f"Backup thành công: {backup_path}"})
    except Exception as e:
        logger.error(f"Error backing up database: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/database/export', methods=['POST'])
def api_export_database():
    """API export database"""
    try:
        export_path = f"face_database_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        success = face_database.export_to_json(export_path)
        if success:
            return jsonify({"success": True, "message": f"Export thành công: {export_path}"})
        else:
            return jsonify({"success": False, "message": "Lỗi export"})
    except Exception as e:
        logger.error(f"Error exporting database: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/recognition/settings', methods=['GET'])
def api_get_recognition_settings():
    """API lấy settings nhận diện"""
    try:
        return jsonify({
            "success": True,
            "settings": recognition_settings,
            "available_methods": ["cosine_similarity", "euclidean_distance", "manhattan_distance"]
        })
    except Exception as e:
        logger.error(f"Error getting recognition settings: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/recognition/settings', methods=['POST'])
def api_update_recognition_settings():
    """API cập nhật settings nhận diện"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "message": "Thiếu dữ liệu"})
        
        # Validate settings
        if 'method' in data:
            if data['method'] not in ["cosine_similarity", "euclidean_distance", "manhattan_distance"]:
                return jsonify({"success": False, "message": "Phương pháp không hợp lệ"})
            recognition_settings['method'] = data['method']
        
        if 'threshold' in data:
            threshold = float(data['threshold'])
            if not (0.0 <= threshold <= 1.0):
                return jsonify({"success": False, "message": "Threshold phải trong khoảng 0.0-1.0"})
            recognition_settings['threshold'] = threshold
        
        if 'unknown_threshold' in data:
            unknown_threshold = float(data['unknown_threshold'])
            if not (0.0 <= unknown_threshold <= 1.0):
                return jsonify({"success": False, "message": "Unknown threshold phải trong khoảng 0.0-1.0"})
            recognition_settings['unknown_threshold'] = unknown_threshold
        
        # Update comparator
        if update_comparator():
            return jsonify({
                "success": True, 
                "message": "Cập nhật settings thành công",
                "settings": recognition_settings
            })
        else:
            return jsonify({"success": False, "message": "Lỗi cập nhật comparator"})
            
    except Exception as e:
        logger.error(f"Error updating recognition settings: {e}")
        return jsonify({"success": False, "message": str(e)})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# Main
if __name__ == '__main__':
    # Initialize models
    if initialize_models():
        logger.info("All models initialized successfully")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize models")
        sys.exit(1)
