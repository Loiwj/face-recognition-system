"""
FaceNet Embedding Module sử dụng facenet-pytorch
"""

import numpy as np
import cv2
import os
import logging
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.error("PyTorch không được cài đặt. Chạy: pip install torch")

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACENET_PYTORCH_AVAILABLE = True
except ImportError:
    FACENET_PYTORCH_AVAILABLE = False
    logging.error("facenet-pytorch không được cài đặt. Chạy: pip install facenet-pytorch")

class FaceNetModel(Enum):
    VGGFACE2 = "vggface2"
    CASIA = "casia-webface"

@dataclass
class FaceNetConfig:
    model_type: FaceNetModel = FaceNetModel.VGGFACE2
    model_path: Optional[str] = None
    embedding_size: int = 512
    l2_normalize: bool = True
    batch_size: int = 32
    use_gpu: bool = True

class FaceNetEmbedder:
    def __init__(self, config: Optional[FaceNetConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch không được cài đặt")
        
        if not FACENET_PYTORCH_AVAILABLE:
            raise ImportError("facenet-pytorch không được cài đặt. Chạy: pip install facenet-pytorch")
        
        self.config = config or FaceNetConfig()
        self.model = None
        self.device = None
        self.logger = logging.getLogger(__name__)
        
        self._setup_device()
        self._load_model()
    
    def _setup_device(self):
        """Setup device (GPU/CPU)"""
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info(f"Sử dụng GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            self.logger.warning("Không tìm thấy GPU, sử dụng CPU")
    
    def _load_model(self):
        """Load FaceNet model"""
        try:
            # Load pre-trained model
            if self.config.model_type == FaceNetModel.VGGFACE2:
                self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            elif self.config.model_type == FaceNetModel.CASIA:
                self.model = InceptionResnetV1(pretrained='casia-webface').eval().to(self.device)
            else:
                raise ValueError(f"Model type không hỗ trợ: {self.config.model_type}")
            
            self.logger.info(f"Đã load FaceNet model: {self.config.model_type.value}")
        except Exception as e:
            self.logger.error(f"Lỗi load model: {e}")
            raise
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding từ ảnh khuôn mặt"""
        try:
            # Convert numpy array to tensor
            if face_image.dtype != np.float32:
                face_image = face_image.astype(np.float32)
            
            # Normalize to [0, 1] if needed
            if face_image.max() > 1.0:
                face_image = face_image / 255.0
            
            # Convert to tensor and add batch dimension
            face_tensor = torch.from_numpy(face_image).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(face_tensor)
                
                # L2 normalize if needed
                if self.config.l2_normalize:
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                
                # Convert back to numpy
                embedding_np = embedding.cpu().numpy().flatten()
                
                return embedding_np
                
        except Exception as e:
            self.logger.error(f"Lỗi extract embedding: {e}")
            return None
    
    def extract_embeddings_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """Extract embeddings cho batch ảnh"""
        if not face_images:
            return []
        
        try:
            embeddings = []
            for face_image in face_images:
                embedding = self.extract_embedding(face_image)
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    embeddings.append(None)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Lỗi extract batch embeddings: {e}")
            return []
    
    def get_embedding_size(self) -> int:
        """Get embedding size"""
        return self.config.embedding_size
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': f'facenet-pytorch-{self.config.model_type.value}',
            'input_shape': (160, 160, 3),
            'embedding_size': self.config.embedding_size,
            'l2_normalize': self.config.l2_normalize,
            'device': str(self.device),
            'model_path': 'facenet-pytorch library'
        }

class FaceNetFactory:
    @staticmethod
    def create_embedder(model_type: str = "vggface2", 
                       model_path: Optional[str] = None,
                       **kwargs) -> 'FaceNetEmbedder':
        """Create FaceNet embedder"""
        if model_type not in ["vggface2", "casia-webface"]:
            raise ValueError("Model type phải là 'vggface2' hoặc 'casia-webface'")
        
        model_enum = FaceNetModel.VGGFACE2 if model_type == "vggface2" else FaceNetModel.CASIA
        config = FaceNetConfig(model_type=model_enum, **kwargs)
        return FaceNetEmbedder(config)
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get available models"""
        if FACENET_PYTORCH_AVAILABLE:
            return ["vggface2", "casia-webface"]
        else:
            return []

# Example usage
if __name__ == "__main__":
    # Test FaceNet embedder
    try:
        embedder = FaceNetFactory.create_embedder("vggface2")
        print("FaceNet embedder created successfully!")
        
        # Test with random image
        test_image = np.random.rand(160, 160, 3).astype(np.float32)
        embedding = embedder.extract_embedding(test_image)
        
        if embedding is not None:
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding norm: {np.linalg.norm(embedding)}")
            print("Test successful!")
        else:
            print("Failed to extract embedding")
            
    except Exception as e:
        print(f"Error: {e}")
