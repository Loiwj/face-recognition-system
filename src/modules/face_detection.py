"""
Module 1: Face Detection
Hỗ trợ nhiều công nghệ phát hiện khuôn mặt: Mediapipe, MTCNN, RetinaFace, YOLOv8
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import logging

# Import các thư viện face detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("Mediapipe không được cài đặt. Sử dụng OpenCV DNN thay thế.")

try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    logging.warning("MTCNN không được cài đặt.")

try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False
    logging.warning("RetinaFace không được cài đặt.")

# OpenCV DNN cho YOLOv8-Face
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch không được cài đặt. YOLOv8-Face sẽ không khả dụng.")


class FaceDetectionResult:
    """Kết quả phát hiện khuôn mặt"""
    
    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float, 
                 landmarks: Optional[List[Tuple[int, int]]] = None):
        """
        Args:
            bbox: (x, y, width, height) - bounding box của khuôn mặt
            confidence: Độ tin cậy (0-1)
            landmarks: Danh sách landmarks (x, y) nếu có
        """
        self.bbox = bbox
        self.confidence = confidence
        self.landmarks = landmarks or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dictionary"""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'landmarks': self.landmarks
        }


class BaseFaceDetector(ABC):
    """Base class cho tất cả face detectors"""
    
    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Phát hiện khuôn mặt trong ảnh
        
        Args:
            image: Ảnh đầu vào (BGR format)
            
        Returns:
            Danh sách kết quả phát hiện khuôn mặt
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Tên của detector"""
        pass


class MediapipeFaceDetector(BaseFaceDetector):
    """Face detector sử dụng Mediapipe - nhanh nhất, tối ưu cho real-time"""
    
    def __init__(self, min_detection_confidence: float = 0.5):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("Mediapipe không được cài đặt. Chạy: pip install mediapipe")
        
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0: 2m-10m range, 1: 5m-50m range
            min_detection_confidence=min_detection_confidence
        )
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Phát hiện khuôn mặt với Mediapipe"""
        # Chuyển BGR sang RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                # Lấy bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Đảm bảo coordinates hợp lệ
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                confidence = detection.score[0]
                faces.append(FaceDetectionResult(
                    bbox=(x, y, width, height),
                    confidence=confidence
                ))
        
        return faces
    
    def get_name(self) -> str:
        return "Mediapipe"


class MTCNNFaceDetector(BaseFaceDetector):
    """Face detector sử dụng MTCNN - cân bằng tốt giữa tốc độ và độ chính xác"""
    
    def __init__(self, min_face_size: int = 20, 
                 scale_factor: float = 0.709,
                 min_confidence: float = 0.5):
        if not MTCNN_AVAILABLE:
            raise ImportError("MTCNN không được cài đặt. Chạy: pip install mtcnn")
        
        self.detector = MTCNN(
            min_face_size=min_face_size,
            scale_factor=scale_factor,
            min_confidence=min_confidence
        )
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Phát hiện khuôn mặt với MTCNN"""
        # MTCNN expects RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb_image)
        
        faces = []
        for result in results:
            if result['confidence'] > 0.5:  # Threshold
                bbox = result['box']
                x, y, w, h = bbox
                
                # Lấy landmarks nếu có
                landmarks = []
                if 'keypoints' in result:
                    for point in result['keypoints'].values():
                        landmarks.append((int(point[0]), int(point[1])))
                
                faces.append(FaceDetectionResult(
                    bbox=(x, y, w, h),
                    confidence=result['confidence'],
                    landmarks=landmarks
                ))
        
        return faces
    
    def get_name(self) -> str:
        return "MTCNN"


class RetinaFaceDetector(BaseFaceDetector):
    """Face detector sử dụng RetinaFace - độ chính xác cao nhất"""
    
    def __init__(self, threshold: float = 0.5):
        if not RETINAFACE_AVAILABLE:
            raise ImportError("RetinaFace không được cài đặt. Chạy: pip install retinaface")
        
        self.threshold = threshold
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Phát hiện khuôn mặt với RetinaFace"""
        # RetinaFace expects RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = RetinaFace.detect_faces(rgb_image, threshold=self.threshold)
        
        faces = []
        if isinstance(results, dict):
            for face_id, face_data in results.items():
                bbox = face_data['facial_area']
                x, y, w, h = bbox
                
                # Lấy landmarks nếu có
                landmarks = []
                if 'landmarks' in face_data:
                    for point in face_data['landmarks'].values():
                        landmarks.append((int(point[0]), int(point[1])))
                
                faces.append(FaceDetectionResult(
                    bbox=(x, y, w, h),
                    confidence=face_data['score'],
                    landmarks=landmarks
                ))
        
        return faces
    
    def get_name(self) -> str:
        return "RetinaFace"


class OpenCVFaceDetector(BaseFaceDetector):
    """Face detector sử dụng OpenCV Haar Cascades - fallback option"""
    
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5):
        # Load Haar cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Phát hiện khuôn mặt với OpenCV Haar Cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            # OpenCV không cung cấp confidence score
            results.append(FaceDetectionResult(
                bbox=(x, y, w, h),
                confidence=0.8  # Default confidence
            ))
        
        return results
    
    def get_name(self) -> str:
        return "OpenCV Haar"


class FaceDetectorFactory:
    """Factory class để tạo face detectors"""
    
    @staticmethod
    def create_detector(detector_type: str, **kwargs) -> BaseFaceDetector:
        """
        Tạo face detector theo loại
        
        Args:
            detector_type: Loại detector ('mediapipe', 'mtcnn', 'retinaface', 'opencv')
            **kwargs: Tham số cho detector
            
        Returns:
            Face detector instance
        """
        detector_type = detector_type.lower()
        
        if detector_type == 'mediapipe' and MEDIAPIPE_AVAILABLE:
            return MediapipeFaceDetector(**kwargs)
        elif detector_type == 'mtcnn' and MTCNN_AVAILABLE:
            return MTCNNFaceDetector(**kwargs)
        elif detector_type == 'retinaface' and RETINAFACE_AVAILABLE:
            return RetinaFaceDetector(**kwargs)
        elif detector_type == 'opencv':
            return OpenCVFaceDetector(**kwargs)
        else:
            # Fallback to OpenCV nếu detector được yêu cầu không khả dụng
            logging.warning(f"Detector {detector_type} không khả dụng. Sử dụng OpenCV thay thế.")
            return OpenCVFaceDetector(**kwargs)
    
    @staticmethod
    def get_available_detectors() -> List[str]:
        """Lấy danh sách các detector khả dụng"""
        available = ['opencv']  # OpenCV luôn khả dụng
        
        if MEDIAPIPE_AVAILABLE:
            available.append('mediapipe')
        if MTCNN_AVAILABLE:
            available.append('mtcnn')
        if RETINAFACE_AVAILABLE:
            available.append('retinaface')
        
        return available


# Utility functions
def draw_faces(image: np.ndarray, faces: List[FaceDetectionResult], 
               show_confidence: bool = True, show_landmarks: bool = True) -> np.ndarray:
    """
    Vẽ bounding box và landmarks lên ảnh
    
    Args:
        image: Ảnh gốc
        faces: Danh sách kết quả phát hiện khuôn mặt
        show_confidence: Hiển thị confidence score
        show_landmarks: Hiển thị landmarks
        
    Returns:
        Ảnh đã được vẽ
    """
    result_image = image.copy()
    
    for i, face in enumerate(faces):
        x, y, w, h = face.bbox
        
        # Vẽ bounding box
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Vẽ confidence score
        if show_confidence:
            label = f"Face {i+1}: {face.confidence:.2f}"
            cv2.putText(result_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Vẽ landmarks
        if show_landmarks and face.landmarks:
            for landmark in face.landmarks:
                cv2.circle(result_image, landmark, 2, (255, 0, 0), -1)
    
    return result_image


def filter_faces_by_confidence(faces: List[FaceDetectionResult], 
                              min_confidence: float = 0.5) -> List[FaceDetectionResult]:
    """Lọc khuôn mặt theo confidence threshold"""
    return [face for face in faces if face.confidence >= min_confidence]


def filter_faces_by_size(faces: List[FaceDetectionResult], 
                        min_size: int = 30) -> List[FaceDetectionResult]:
    """Lọc khuôn mặt theo kích thước tối thiểu"""
    return [face for face in faces 
            if face.bbox[2] >= min_size and face.bbox[3] >= min_size]


# Example usage
if __name__ == "__main__":
    # Test với webcam
    cap = cv2.VideoCapture(0)
    
    # Tạo detector
    detector = FaceDetectorFactory.create_detector('mediapipe')
    
    print(f"Sử dụng detector: {detector.get_name()}")
    print("Nhấn 'q' để thoát")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Phát hiện khuôn mặt
        faces = detector.detect_faces(frame)
        
        # Vẽ kết quả
        result_frame = draw_faces(frame, faces)
        
        # Hiển thị
        cv2.imshow('Face Detection', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
