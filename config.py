"""
Configuration file cho hệ thống nhận diện khuôn mặt
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class FaceDetectionConfig:
    """Cấu hình face detection"""
    detector_type: str = "mediapipe"  # mediapipe, mtcnn, retinaface, opencv
    min_confidence: float = 0.5
    min_face_size: int = 32


@dataclass
class PreprocessingConfig:
    """Cấu hình preprocessing"""
    target_size: tuple = (160, 160)
    margin: float = 0.2
    normalize_method: str = "standard"  # standard, minmax, none
    apply_alignment: bool = True
    quality_check: bool = True


@dataclass
class FaceNetConfig:
    """Cấu hình FaceNet"""
    model_type: str = "keras_facenet"  # Chỉ hỗ trợ keras_facenet
    model_path: Optional[str] = None  # Không sử dụng với keras-facenet
    embedding_size: int = 512
    l2_normalize: bool = True
    use_gpu: bool = True


@dataclass
class ComparisonConfig:
    """Cấu hình comparison"""
    method: str = "cosine_similarity"  # cosine_similarity, euclidean_distance, svm_classifier, knn_classifier
    threshold: float = 0.6
    unknown_threshold: float = 0.4
    top_k: int = 1


@dataclass
class DatabaseConfig:
    """Cấu hình database"""
    db_path: str = "face_database.db"
    backup_path: str = "face_database.db.backup"
    auto_backup: bool = True
    backup_interval: int = 24  # hours


@dataclass
class WebConfig:
    """Cấu hình web application"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = True
    secret_key: str = "your-secret-key-here"
    upload_folder: str = "static/uploads"
    max_file_size: int = 16 * 1024 * 1024  # 16MB
    allowed_extensions: set = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


@dataclass
class TrainingConfig:
    """Cấu hình training"""
    dataset_path: str = "data/training"
    output_dir: str = "models/trained"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5


@dataclass
class SystemConfig:
    """Cấu hình hệ thống"""
    # Face Detection
    face_detection: FaceDetectionConfig = FaceDetectionConfig()
    
    # Preprocessing
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    
    # FaceNet
    facenet: FaceNetConfig = FaceNetConfig()
    
    # Comparison
    comparison: ComparisonConfig = ComparisonConfig()
    
    # Database
    database: DatabaseConfig = DatabaseConfig()
    
    # Web Application
    web: WebConfig = WebConfig()
    
    # Training
    training: TrainingConfig = TrainingConfig()
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    def __post_init__(self):
        # Create necessary directories
        os.makedirs(self.web.upload_folder, exist_ok=True)
        os.makedirs(self.training.output_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("data/training", exist_ok=True)
        os.makedirs("data/test", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'face_detection': self.face_detection.__dict__,
            'preprocessing': self.preprocessing.__dict__,
            'facenet': self.facenet.__dict__,
            'comparison': self.comparison.__dict__,
            'database': self.database.__dict__,
            'web': self.web.__dict__,
            'training': self.training.__dict__,
            'log_level': self.log_level,
            'log_file': self.log_file
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """Create config from dictionary"""
        config = cls()
        
        if 'face_detection' in config_dict:
            config.face_detection = FaceDetectionConfig(**config_dict['face_detection'])
        
        if 'preprocessing' in config_dict:
            config.preprocessing = PreprocessingConfig(**config_dict['preprocessing'])
        
        if 'facenet' in config_dict:
            config.facenet = FaceNetConfig(**config_dict['facenet'])
        
        if 'comparison' in config_dict:
            config.comparison = ComparisonConfig(**config_dict['comparison'])
        
        if 'database' in config_dict:
            config.database = DatabaseConfig(**config_dict['database'])
        
        if 'web' in config_dict:
            config.web = WebConfig(**config_dict['web'])
        
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        
        if 'log_level' in config_dict:
            config.log_level = config_dict['log_level']
        
        if 'log_file' in config_dict:
            config.log_file = config_dict['log_file']
        
        return config


# Default configuration
DEFAULT_CONFIG = SystemConfig()

# Environment-specific configurations
DEVELOPMENT_CONFIG = SystemConfig(
    web=WebConfig(debug=True, host="127.0.0.1"),
    log_level="DEBUG"
)

PRODUCTION_CONFIG = SystemConfig(
    web=WebConfig(debug=False, host="0.0.0.0"),
    log_level="WARNING",
    database=DatabaseConfig(auto_backup=True, backup_interval=12)
)

# Load configuration from environment
def load_config() -> SystemConfig:
    """Load configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    
    if env == 'production':
        return PRODUCTION_CONFIG
    elif env == 'development':
        return DEVELOPMENT_CONFIG
    else:
        return DEFAULT_CONFIG


# Export default config
config = load_config()
