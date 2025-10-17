"""
Module 2: Image Preprocessing
Chuẩn hóa ảnh mặt để đảm bảo tính nhất quán cho mô hình FaceNet
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

from .face_detection import FaceDetectionResult


class PreprocessingMode(Enum):
    """Chế độ preprocessing"""
    INFERENCE = "inference"  # Cho nhận diện
    TRAINING = "training"    # Cho training với augmentation
    ENROLLMENT = "enrollment"  # Cho đăng ký


@dataclass
class PreprocessingConfig:
    """Cấu hình preprocessing"""
    target_size: Tuple[int, int] = (160, 160)  # Kích thước chuẩn FaceNet
    margin: float = 0.2  # Margin cho crop (20%)
    normalize_method: str = "standard"  # "standard", "minmax", "none"
    apply_alignment: bool = True
    quality_check: bool = True
    min_face_size: int = 32
    augmentation: bool = False  # Chỉ dùng cho training


class FacePreprocessor:
    """Class chính cho preprocessing ảnh khuôn mặt"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(__name__)
    
    def preprocess_face(self, image: np.ndarray, face_result: FaceDetectionResult,
                       mode: PreprocessingMode = PreprocessingMode.INFERENCE) -> Optional[np.ndarray]:
        """
        Preprocessing một khuôn mặt
        
        Args:
            image: Ảnh gốc (BGR format)
            face_result: Kết quả phát hiện khuôn mặt
            mode: Chế độ preprocessing
            
        Returns:
            Ảnh đã được preprocessing hoặc None nếu không hợp lệ
        """
        try:
            # 1. Crop khuôn mặt
            cropped_face = self._crop_face(image, face_result)
            if cropped_face is None:
                return None
            
            # 2. Quality check
            if self.config.quality_check and not self._quality_check(cropped_face):
                self.logger.warning("Khuôn mặt không đạt chất lượng")
                return None
            
            # 3. Alignment (nếu có landmarks)
            if self.config.apply_alignment and face_result.landmarks:
                cropped_face = self._align_face(cropped_face, face_result.landmarks)
            
            # 4. Resize về kích thước chuẩn
            resized_face = self._resize_face(cropped_face)
            
            # 5. Normalization
            normalized_face = self._normalize_face(resized_face)
            
            # 6. Data augmentation (chỉ cho training)
            if mode == PreprocessingMode.TRAINING and self.config.augmentation:
                normalized_face = self._apply_augmentation(normalized_face)
            
            return normalized_face
            
        except Exception as e:
            self.logger.error(f"Lỗi preprocessing: {e}")
            return None
    
    def preprocess_faces(self, image: np.ndarray, face_results: List[FaceDetectionResult],
                        mode: PreprocessingMode = PreprocessingMode.INFERENCE) -> List[np.ndarray]:
        """
        Preprocessing nhiều khuôn mặt
        
        Args:
            image: Ảnh gốc
            face_results: Danh sách kết quả phát hiện khuôn mặt
            mode: Chế độ preprocessing
            
        Returns:
            Danh sách ảnh đã được preprocessing
        """
        processed_faces = []
        
        for face_result in face_results:
            processed_face = self.preprocess_face(image, face_result, mode)
            if processed_face is not None:
                processed_faces.append(processed_face)
        
        return processed_faces
    
    def _crop_face(self, image: np.ndarray, face_result: FaceDetectionResult) -> Optional[np.ndarray]:
        """Crop khuôn mặt với margin"""
        x, y, w, h = face_result.bbox
        img_h, img_w = image.shape[:2]
        
        # Tính margin
        margin_x = int(w * self.config.margin)
        margin_y = int(h * self.config.margin)
        
        # Tính tọa độ crop
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + w + margin_x)
        y2 = min(img_h, y + h + margin_y)
        
        # Crop
        cropped = image[y1:y2, x1:x2]
        
        if cropped.size == 0:
            return None
        
        return cropped
    
    def _quality_check(self, face_image: np.ndarray) -> bool:
        """Kiểm tra chất lượng khuôn mặt"""
        h, w = face_image.shape[:2]
        
        # Kiểm tra kích thước tối thiểu
        if h < self.config.min_face_size or w < self.config.min_face_size:
            return False
        
        # Kiểm tra độ sáng
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Quá tối hoặc quá sáng
        if mean_brightness < 30 or mean_brightness > 220:
            return False
        
        # Kiểm tra độ tương phản
        contrast = np.std(gray)
        if contrast < 20:  # Độ tương phản quá thấp
            return False
        
        return True
    
    def _align_face(self, face_image: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """
        Alignment khuôn mặt dựa trên landmarks
        Sử dụng 5-point alignment (2 mắt, mũi, 2 góc miệng)
        """
        if len(landmarks) < 5:
            return face_image
        
        # Định nghĩa landmarks chuẩn cho khuôn mặt 160x160
        # (mắt trái, mắt phải, mũi, góc miệng trái, góc miệng phải)
        standard_landmarks = np.array([
            [38.2946, 51.6963],   # Mắt trái
            [73.5318, 51.5014],   # Mắt phải
            [56.0252, 71.7366],   # Mũi
            [41.5493, 92.3655],   # Góc miệng trái
            [70.7299, 92.2041]    # Góc miệng phải
        ], dtype=np.float32)
        
        # Lấy 5 landmarks từ kết quả detection
        # Giả sử landmarks được sắp xếp theo thứ tự: mắt trái, mắt phải, mũi, miệng trái, miệng phải
        if len(landmarks) >= 5:
            detected_landmarks = np.array(landmarks[:5], dtype=np.float32)
        else:
            # Nếu không đủ landmarks, sử dụng center alignment
            return self._center_align(face_image)
        
        # Tính transformation matrix
        transform_matrix = cv2.getAffineTransform(
            detected_landmarks[:3],  # 3 điểm đầu
            standard_landmarks[:3]   # 3 điểm chuẩn
        )
        
        # Áp dụng transformation
        aligned_face = cv2.warpAffine(
            face_image, transform_matrix, 
            (self.config.target_size[0], self.config.target_size[1]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        return aligned_face
    
    def _center_align(self, face_image: np.ndarray) -> np.ndarray:
        """Center alignment khi không có landmarks"""
        h, w = face_image.shape[:2]
        
        # Tính tỷ lệ scale
        scale = min(self.config.target_size[0] / w, self.config.target_size[1] / h)
        
        # Resize với giữ nguyên tỷ lệ
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(face_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Tạo ảnh đích với padding
        result = np.zeros((self.config.target_size[1], self.config.target_size[0], 3), dtype=np.uint8)
        
        # Tính offset để center
        y_offset = (self.config.target_size[1] - new_h) // 2
        x_offset = (self.config.target_size[0] - new_w) // 2
        
        # Paste ảnh đã resize vào center
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return result
    
    def _resize_face(self, face_image: np.ndarray) -> np.ndarray:
        """Resize khuôn mặt về kích thước chuẩn"""
        return cv2.resize(
            face_image, 
            self.config.target_size, 
            interpolation=cv2.INTER_LINEAR
        )
    
    def _normalize_face(self, face_image: np.ndarray) -> np.ndarray:
        """Normalization ảnh khuôn mặt"""
        if self.config.normalize_method == "standard":
            # Chuẩn hóa về [0, 1] rồi standardize
            normalized = face_image.astype(np.float32) / 255.0
            # Mean subtraction (có thể dùng mean của dataset)
            normalized = normalized - 0.5
            # Standardization
            normalized = normalized / 0.5
            return normalized
        
        elif self.config.normalize_method == "minmax":
            # Min-max normalization về [0, 1]
            normalized = face_image.astype(np.float32) / 255.0
            return normalized
        
        else:  # "none"
            return face_image.astype(np.float32)
    
    def _apply_augmentation(self, face_image: np.ndarray) -> np.ndarray:
        """Áp dụng data augmentation cho training"""
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = face_image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            face_image = cv2.warpAffine(face_image, rotation_matrix, (w, h))
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            face_image = cv2.flip(face_image, 1)
        
        # Random brightness/contrast
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contrast
            beta = np.random.uniform(-20, 20)    # Brightness
            face_image = cv2.convertScaleAbs(face_image, alpha=alpha, beta=beta)
        
        # Random noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 5, face_image.shape).astype(np.float32)
            face_image = np.clip(face_image + noise, 0, 255).astype(np.uint8)
        
        return face_image


class BatchPreprocessor:
    """Preprocessor cho batch processing"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.preprocessor = FacePreprocessor(config)
    
    def preprocess_batch(self, images: List[np.ndarray], 
                        face_results_list: List[List[FaceDetectionResult]],
                        mode: PreprocessingMode = PreprocessingMode.INFERENCE) -> List[List[np.ndarray]]:
        """
        Preprocessing batch ảnh
        
        Args:
            images: Danh sách ảnh
            face_results_list: Danh sách kết quả phát hiện khuôn mặt cho mỗi ảnh
            mode: Chế độ preprocessing
            
        Returns:
            Danh sách ảnh đã được preprocessing cho mỗi ảnh gốc
        """
        results = []
        
        for image, face_results in zip(images, face_results_list):
            processed_faces = self.preprocessor.preprocess_faces(image, face_results, mode)
            results.append(processed_faces)
        
        return results


# Utility functions
def create_preprocessing_config(mode: PreprocessingMode, 
                              target_size: Tuple[int, int] = (160, 160),
                              **kwargs) -> PreprocessingConfig:
    """Tạo config preprocessing theo mode"""
    config = PreprocessingConfig(target_size=target_size)
    
    if mode == PreprocessingMode.TRAINING:
        config.augmentation = True
        config.quality_check = True
        config.apply_alignment = True
    elif mode == PreprocessingMode.ENROLLMENT:
        config.augmentation = False
        config.quality_check = True
        config.apply_alignment = True
    else:  # INFERENCE
        config.augmentation = False
        config.quality_check = False
        config.apply_alignment = True
    
    # Override với kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def visualize_preprocessing_steps(image: np.ndarray, face_result: FaceDetectionResult,
                                config: Optional[PreprocessingConfig] = None) -> List[np.ndarray]:
    """
    Visualize các bước preprocessing để debug
    
    Returns:
        Danh sách ảnh ở các bước khác nhau
    """
    preprocessor = FacePreprocessor(config)
    steps = []
    
    # Bước 1: Crop
    cropped = preprocessor._crop_face(image, face_result)
    if cropped is not None:
        steps.append(cropped.copy())
        
        # Bước 2: Alignment
        if config.apply_alignment and face_result.landmarks:
            aligned = preprocessor._align_face(cropped, face_result.landmarks)
            steps.append(aligned.copy())
        else:
            steps.append(cropped.copy())
        
        # Bước 3: Resize
        resized = preprocessor._resize_face(steps[-1])
        steps.append(resized.copy())
        
        # Bước 4: Normalization
        normalized = preprocessor._normalize_face(resized)
        # Convert back to uint8 for visualization
        if normalized.dtype == np.float32:
            normalized = (normalized * 255).astype(np.uint8)
        steps.append(normalized.copy())
    
    return steps


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from face_detection import FaceDetectorFactory
    
    # Test preprocessing
    image = cv2.imread("test_image.jpg")
    if image is None:
        print("Không thể đọc ảnh test")
        exit()
    
    # Phát hiện khuôn mặt
    detector = FaceDetectorFactory.create_detector('mediapipe')
    faces = detector.detect_faces(image)
    
    if faces:
        # Preprocessing
        config = create_preprocessing_config(PreprocessingMode.INFERENCE)
        preprocessor = FacePreprocessor(config)
        
        processed_face = preprocessor.preprocess_face(image, faces[0])
        
        if processed_face is not None:
            # Hiển thị kết quả
            cv2.imshow("Original", image)
            cv2.imshow("Processed", processed_face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Preprocessing thất bại")
    else:
        print("Không phát hiện được khuôn mặt")
