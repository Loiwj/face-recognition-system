"""
Module 4: Comparison & Classification
So khớp embedding mới với cơ sở dữ liệu để xác định danh tính
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

try:
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn không được cài đặt. Một số tính năng sẽ không khả dụng.")


class ComparisonMethod(Enum):
    """Các phương pháp so sánh"""
    COSINE_SIMILARITY = "cosine_similarity"
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    MANHATTAN_DISTANCE = "manhattan_distance"
    SVM_CLASSIFIER = "svm_classifier"
    KNN_CLASSIFIER = "knn_classifier"


@dataclass
class ComparisonResult:
    """Kết quả so sánh"""
    identity: str
    confidence: float
    method: str
    distance: Optional[float] = None
    is_unknown: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'identity': self.identity,
            'confidence': float(self.confidence) if self.confidence is not None else None,
            'method': self.method,
            'distance': float(self.distance) if self.distance is not None else None,
            'is_unknown': bool(self.is_unknown)  # Convert numpy bool_ to Python bool
        }


@dataclass
class ComparisonConfig:
    """Cấu hình comparison"""
    method: ComparisonMethod = ComparisonMethod.COSINE_SIMILARITY
    threshold: float = 0.6
    unknown_threshold: float = 0.85
    top_k: int = 1
    use_svm: bool = False
    svm_kernel: str = 'rbf'
    knn_neighbors: int = 3


class BaseComparator(ABC):
    """Base class cho tất cả comparators"""
    
    @abstractmethod
    def compare(self, query_embedding: np.ndarray, 
                database_embeddings: Dict[str, List[np.ndarray]]) -> ComparisonResult:
        """So sánh embedding với database"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Tên của comparator"""
        pass


class CosineSimilarityComparator(BaseComparator):
    """Comparator sử dụng Cosine Similarity"""
    
    def __init__(self, threshold: float = 0.6, unknown_threshold: float = 0.85):
        self.threshold = threshold
        self.unknown_threshold = unknown_threshold
        self.logger = logging.getLogger(__name__)
    
    def compare(self, query_embedding: np.ndarray, 
                database_embeddings: Dict[str, List[np.ndarray]]) -> ComparisonResult:
        """So sánh bằng cosine similarity"""
        if not database_embeddings:
            return ComparisonResult(
                identity="Unknown",
                confidence=0.0,
                method="cosine_similarity",
                is_unknown=True
            )
        
        best_identity = "Unknown"
        best_confidence = 0.0
        best_distance = float('inf')
        
        for identity, embeddings in database_embeddings.items():
            # Tính similarity với tất cả embeddings của identity này
            similarities = []
            for embedding in embeddings:
                similarity = self._cosine_similarity(query_embedding, embedding)
                similarities.append(similarity)
            
            # Lấy similarity cao nhất
            max_similarity = max(similarities)
            
            if max_similarity > best_confidence:
                best_confidence = max_similarity
                best_identity = identity
                best_distance = 1 - max_similarity  # Convert to distance
        
        # Xác định có phải unknown không
        is_unknown = best_confidence < self.unknown_threshold
        
        return ComparisonResult(
            identity=best_identity if not is_unknown else "Unknown",
            confidence=best_confidence,
            method="cosine_similarity",
            distance=best_distance,
            is_unknown=is_unknown
        )
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Tính cosine similarity"""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def get_name(self) -> str:
        return "Cosine Similarity"


class EuclideanDistanceComparator(BaseComparator):
    """Comparator sử dụng Euclidean Distance"""
    
    def __init__(self, threshold: float = 1.0, unknown_threshold: float = 1.5):
        self.threshold = threshold
        self.unknown_threshold = unknown_threshold
        self.logger = logging.getLogger(__name__)
    
    def compare(self, query_embedding: np.ndarray, 
                database_embeddings: Dict[str, List[np.ndarray]]) -> ComparisonResult:
        """So sánh bằng euclidean distance"""
        if not database_embeddings:
            return ComparisonResult(
                identity="Unknown",
                confidence=0.0,
                method="euclidean_distance",
                distance=float('inf'),
                is_unknown=True
            )
        
        best_identity = "Unknown"
        best_distance = float('inf')
        
        for identity, embeddings in database_embeddings.items():
            # Tính distance với tất cả embeddings của identity này
            distances = []
            for embedding in embeddings:
                distance = np.linalg.norm(query_embedding - embedding)
                distances.append(distance)
            
            # Lấy distance nhỏ nhất
            min_distance = min(distances)
            
            if min_distance < best_distance:
                best_distance = min_distance
                best_identity = identity
        
        # Convert distance to confidence (0-1)
        confidence = max(0, 1 - (best_distance / self.unknown_threshold))
        
        # Xác định có phải unknown không
        is_unknown = best_distance > self.unknown_threshold
        
        return ComparisonResult(
            identity=best_identity if not is_unknown else "Unknown",
            confidence=confidence,
            method="euclidean_distance",
            distance=best_distance,
            is_unknown=is_unknown
        )
    
    def get_name(self) -> str:
        return "Euclidean Distance"


class ManhattanDistanceComparator(BaseComparator):
    """Comparator sử dụng Manhattan Distance"""
    
    def __init__(self, threshold: float = 1.0, unknown_threshold: float = 1.5):
        self.threshold = threshold
        self.unknown_threshold = unknown_threshold
        self.logger = logging.getLogger(__name__)
    
    def compare(self, query_embedding: np.ndarray, 
                database_embeddings: Dict[str, List[np.ndarray]]) -> ComparisonResult:
        """So sánh bằng manhattan distance"""
        if not database_embeddings:
            return ComparisonResult(
                identity="Unknown",
                confidence=0.0,
                method="manhattan_distance",
                distance=float('inf'),
                is_unknown=True
            )
        
        best_identity = "Unknown"
        best_distance = float('inf')
        
        for identity, embeddings in database_embeddings.items():
            # Tính distance với tất cả embeddings của identity này
            distances = []
            for embedding in embeddings:
                distance = np.sum(np.abs(query_embedding - embedding))
                distances.append(distance)
            
            # Lấy distance nhỏ nhất
            min_distance = min(distances)
            
            if min_distance < best_distance:
                best_distance = min_distance
                best_identity = identity
        
        # Convert distance to confidence (0-1)
        confidence = max(0, 1 - (best_distance / self.unknown_threshold))
        
        # Xác định có phải unknown không
        is_unknown = best_distance > self.unknown_threshold
        
        return ComparisonResult(
            identity=best_identity if not is_unknown else "Unknown",
            confidence=confidence,
            method="manhattan_distance",
            distance=best_distance,
            is_unknown=is_unknown
        )
    
    def get_name(self) -> str:
        return "Manhattan Distance"


class SVMComparator(BaseComparator):
    """Comparator sử dụng SVM Classifier"""
    
    def __init__(self, kernel: str = 'rbf', threshold: float = 0.5):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn không được cài đặt")
        
        self.kernel = kernel
        self.threshold = threshold
        self.svm_model = None
        self.scaler = StandardScaler()
        self.identity_to_label = {}
        self.label_to_identity = {}
        self.logger = logging.getLogger(__name__)
    
    def train(self, database_embeddings: Dict[str, List[np.ndarray]]):
        """Train SVM model"""
        if not database_embeddings:
            raise ValueError("Database embeddings trống")
        
        # Chuẩn bị data
        X = []
        y = []
        
        for identity, embeddings in database_embeddings.items():
            if identity not in self.identity_to_label:
                label = len(self.identity_to_label)
                self.identity_to_label[identity] = label
                self.label_to_identity[label] = identity
            
            label = self.identity_to_label[identity]
            for embedding in embeddings:
                X.append(embedding)
                y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train SVM
        self.svm_model = SVC(kernel=self.kernel, probability=True, random_state=42)
        self.svm_model.fit(X_scaled, y)
        
        self.logger.info(f"Đã train SVM với {len(self.identity_to_label)} identities")
    
    def compare(self, query_embedding: np.ndarray, 
                database_embeddings: Dict[str, List[np.ndarray]]) -> ComparisonResult:
        """So sánh bằng SVM"""
        if self.svm_model is None:
            # Train nếu chưa train
            self.train(database_embeddings)
        
        # Scale query embedding
        query_scaled = self.scaler.transform([query_embedding])
        
        # Predict
        prediction = self.svm_model.predict(query_scaled)[0]
        probabilities = self.svm_model.predict_proba(query_scaled)[0]
        
        # Lấy confidence cao nhất
        max_confidence = np.max(probabilities)
        predicted_identity = self.label_to_identity.get(prediction, "Unknown")
        
        # Xác định có phải unknown không
        is_unknown = max_confidence < self.threshold
        
        return ComparisonResult(
            identity=predicted_identity if not is_unknown else "Unknown",
            confidence=max_confidence,
            method="svm_classifier",
            is_unknown=is_unknown
        )
    
    def get_name(self) -> str:
        return f"SVM ({self.kernel})"


class KNNComparator(BaseComparator):
    """Comparator sử dụng K-Nearest Neighbors"""
    
    def __init__(self, n_neighbors: int = 3, threshold: float = 0.5):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn không được cài đặt")
        
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.knn_model = None
        self.scaler = StandardScaler()
        self.identity_to_label = {}
        self.label_to_identity = {}
        self.logger = logging.getLogger(__name__)
    
    def train(self, database_embeddings: Dict[str, List[np.ndarray]]):
        """Train KNN model"""
        if not database_embeddings:
            raise ValueError("Database embeddings trống")
        
        # Chuẩn bị data
        X = []
        y = []
        
        for identity, embeddings in database_embeddings.items():
            if identity not in self.identity_to_label:
                label = len(self.identity_to_label)
                self.identity_to_label[identity] = label
                self.label_to_identity[label] = identity
            
            label = self.identity_to_label[identity]
            for embedding in embeddings:
                X.append(embedding)
                y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train KNN
        self.knn_model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn_model.fit(X_scaled, y)
        
        self.logger.info(f"Đã train KNN với {len(self.identity_to_label)} identities")
    
    def compare(self, query_embedding: np.ndarray, 
                database_embeddings: Dict[str, List[np.ndarray]]) -> ComparisonResult:
        """So sánh bằng KNN"""
        if self.knn_model is None:
            # Train nếu chưa train
            self.train(database_embeddings)
        
        # Scale query embedding
        query_scaled = self.scaler.transform([query_embedding])
        
        # Predict
        prediction = self.knn_model.predict(query_scaled)[0]
        probabilities = self.knn_model.predict_proba(query_scaled)[0]
        
        # Lấy confidence cao nhất
        max_confidence = np.max(probabilities)
        predicted_identity = self.label_to_identity.get(prediction, "Unknown")
        
        # Xác định có phải unknown không
        is_unknown = max_confidence < self.threshold
        
        return ComparisonResult(
            identity=predicted_identity if not is_unknown else "Unknown",
            confidence=max_confidence,
            method="knn_classifier",
            is_unknown=is_unknown
        )
    
    def get_name(self) -> str:
        return f"KNN (k={self.n_neighbors})"


class FaceComparator:
    """Class chính để so sánh khuôn mặt"""
    
    def __init__(self, config: Optional[ComparisonConfig] = None):
        self.config = config or ComparisonConfig()
        self.comparator = self._create_comparator()
        self.logger = logging.getLogger(__name__)
    
    def _create_comparator(self) -> BaseComparator:
        """Tạo comparator theo config"""
        if self.config.method == ComparisonMethod.COSINE_SIMILARITY:
            return CosineSimilarityComparator(
                threshold=self.config.threshold,
                unknown_threshold=self.config.unknown_threshold
            )
        elif self.config.method == ComparisonMethod.EUCLIDEAN_DISTANCE:
            return EuclideanDistanceComparator(
                threshold=self.config.threshold,
                unknown_threshold=self.config.unknown_threshold
            )
        elif self.config.method == ComparisonMethod.MANHATTAN_DISTANCE:
            return ManhattanDistanceComparator(
                threshold=self.config.threshold,
                unknown_threshold=self.config.unknown_threshold
            )
        elif self.config.method == ComparisonMethod.SVM_CLASSIFIER:
            return SVMComparator(
                kernel=self.config.svm_kernel,
                threshold=self.config.threshold
            )
        elif self.config.method == ComparisonMethod.KNN_CLASSIFIER:
            return KNNComparator(
                n_neighbors=self.config.knn_neighbors,
                threshold=self.config.threshold
            )
        else:
            raise ValueError(f"Method không được hỗ trợ: {self.config.method}")
    
    def compare_face(self, query_embedding: np.ndarray, 
                    database_embeddings: Dict[str, List[np.ndarray]]) -> ComparisonResult:
        """So sánh một khuôn mặt"""
        return self.comparator.compare(query_embedding, database_embeddings)
    
    def compare_faces_batch(self, query_embeddings: List[np.ndarray], 
                           database_embeddings: Dict[str, List[np.ndarray]]) -> List[ComparisonResult]:
        """So sánh batch khuôn mặt"""
        results = []
        for embedding in query_embeddings:
            result = self.comparator.compare(embedding, database_embeddings)
            results.append(result)
        return results
    
    def train_classifier(self, database_embeddings: Dict[str, List[np.ndarray]]):
        """Train classifier (cho SVM và KNN)"""
        if hasattr(self.comparator, 'train'):
            self.comparator.train(database_embeddings)
    
    def get_comparator_name(self) -> str:
        """Lấy tên comparator"""
        return self.comparator.get_name()


class ComparisonFactory:
    """Factory class để tạo comparators"""
    
    @staticmethod
    def create_comparator(method: str, **kwargs) -> FaceComparator:
        """
        Tạo comparator theo method
        
        Args:
            method: Phương pháp so sánh
            **kwargs: Tham số cho config
            
        Returns:
            FaceComparator instance
        """
        try:
            method_enum = ComparisonMethod(method)
        except ValueError:
            raise ValueError(f"Method không hợp lệ: {method}")
        
        config = ComparisonConfig(method=method_enum, **kwargs)
        return FaceComparator(config)
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Lấy danh sách methods khả dụng"""
        methods = [
            "cosine_similarity",
            "euclidean_distance",
            "manhattan_distance"
        ]
        
        if SKLEARN_AVAILABLE:
            methods.extend([
                "svm_classifier",
                "knn_classifier"
            ])
        
        return methods


# Utility functions
def calculate_threshold_from_data(embeddings: Dict[str, List[np.ndarray]], 
                                method: str = "cosine_similarity",
                                percentile: float = 95) -> float:
    """
    Tính threshold từ dữ liệu training
    
    Args:
        embeddings: Database embeddings
        method: Phương pháp so sánh
        percentile: Percentile để tính threshold
        
    Returns:
        Threshold value
    """
    similarities = []
    
    for identity, identity_embeddings in embeddings.items():
        # So sánh embeddings của cùng một identity
        for i in range(len(identity_embeddings)):
            for j in range(i + 1, len(identity_embeddings)):
                if method == "cosine_similarity":
                    sim = np.dot(identity_embeddings[i], identity_embeddings[j]) / (
                        np.linalg.norm(identity_embeddings[i]) * np.linalg.norm(identity_embeddings[j])
                    )
                    similarities.append(sim)
                elif method == "euclidean_distance":
                    dist = np.linalg.norm(identity_embeddings[i] - identity_embeddings[j])
                    similarities.append(dist)
    
    if not similarities:
        return 0.6 if method == "cosine_similarity" else 1.0
    
    return np.percentile(similarities, percentile)


def evaluate_comparator(comparator: FaceComparator, 
                       test_embeddings: Dict[str, List[np.ndarray]],
                       ground_truth: List[Tuple[np.ndarray, str]]) -> Dict[str, float]:
    """
    Đánh giá hiệu suất comparator
    
    Args:
        comparator: Comparator để đánh giá
        test_embeddings: Database embeddings
        ground_truth: Danh sách (embedding, true_identity)
        
    Returns:
        Dictionary chứa các metrics
    """
    predictions = []
    true_labels = []
    
    for query_embedding, true_identity in ground_truth:
        result = comparator.compare_face(query_embedding, test_embeddings)
        predictions.append(result.identity)
        true_labels.append(true_identity)
    
    # Tính accuracy
    correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    accuracy = correct / len(predictions)
    
    # Tính precision, recall, f1-score nếu có sklearn
    metrics = {'accuracy': accuracy}
    
    if SKLEARN_AVAILABLE:
        from sklearn.metrics import precision_recall_fscore_support, classification_report
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
    
    return metrics


# Example usage
if __name__ == "__main__":
    # Test comparison
    np.random.seed(42)
    
    # Tạo dữ liệu test
    database_embeddings = {
        "person1": [np.random.randn(512) for _ in range(3)],
        "person2": [np.random.randn(512) for _ in range(3)],
        "person3": [np.random.randn(512) for _ in range(3)]
    }
    
    # Normalize embeddings
    for identity in database_embeddings:
        for i, embedding in enumerate(database_embeddings[identity]):
            database_embeddings[identity][i] = embedding / np.linalg.norm(embedding)
    
    # Test query
    query_embedding = database_embeddings["person1"][0] + np.random.randn(512) * 0.1
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Test các methods
    methods = ["cosine_similarity", "euclidean_distance"]
    
    if SKLEARN_AVAILABLE:
        methods.extend(["svm_classifier", "knn_classifier"])
    
    for method in methods:
        print(f"\nTesting {method}:")
        comparator = ComparisonFactory.create_comparator(method)
        
        if method in ["svm_classifier", "knn_classifier"]:
            comparator.train_classifier(database_embeddings)
        
        result = comparator.compare_face(query_embedding, database_embeddings)
        print(f"Result: {result.to_dict()}")
