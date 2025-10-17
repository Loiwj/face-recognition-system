"""
Evaluation script cho FaceNet model
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import json
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.face_detection import FaceDetectorFactory
from modules.preprocessing import FacePreprocessor, create_preprocessing_config, PreprocessingMode
from modules.facenet_embedding import FaceNetFactory, FaceNetConfig
from modules.comparison import ComparisonFactory, ComparisonMethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class FaceNetEvaluator:
    """Class để đánh giá FaceNet model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.face_detector = None
        self.preprocessor = None
        self.comparator = None
        
        # Initialize components
        self.face_detector = FaceDetectorFactory.create_detector('mtcnn')
        self.preprocessor = FacePreprocessor(create_preprocessing_config(PreprocessingMode.INFERENCE))
        self.comparator = ComparisonFactory.create_comparator("cosine_similarity", threshold=0.6)
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        logger.info("FaceNetEvaluator initialized")
    
    def load_model(self, model_path: str):
        """Load model for evaluation"""
        logger.info(f"Loading model from {model_path}")
        
        try:
            self.model = keras.models.load_model(model_path, compile=False)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_test_data(self, test_data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load test data
        
        Args:
            test_data_path: Đường dẫn đến test data
            
        Returns:
            Tuple (images, labels, label_names)
        """
        logger.info(f"Loading test data from {test_data_path}")
        
        images = []
        labels = []
        label_names = []
        label_to_id = {}
        current_id = 0
        
        # Walk through test data directory
        for root, dirs, files in os.walk(test_data_path):
            if files:  # If there are files in this directory
                # Get person name from directory name
                person_name = os.path.basename(root)
                
                if person_name not in label_to_id:
                    label_to_id[person_name] = current_id
                    current_id += 1
                
                person_id = label_to_id[person_name]
                
                # Process each image
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_path = os.path.join(root, file)
                        
                        try:
                            # Load image
                            image = cv2.imread(image_path)
                            if image is None:
                                continue
                            
                            # Detect faces
                            faces = self.face_detector.detect_faces(image)
                            
                            if faces:
                                # Process first face
                                processed_faces = self.preprocessor.preprocess_faces(
                                    image, faces, PreprocessingMode.INFERENCE
                                )
                                
                                if processed_faces:
                                    images.append(processed_faces[0])
                                    labels.append(person_id)
                                    label_names.append(person_name)
                                    
                        except Exception as e:
                            logger.warning(f"Error processing {image_path}: {e}")
                            continue
        
        if not images:
            raise ValueError("No valid images found in test data")
        
        images = np.array(images)
        labels = np.array(labels)
        
        logger.info(f"Loaded {len(images)} test images for {len(label_to_id)} identities")
        logger.info(f"Image shape: {images.shape}")
        
        return images, labels, label_names
    
    def extract_embeddings(self, images: np.ndarray) -> np.ndarray:
        """Extract embeddings from images"""
        logger.info("Extracting embeddings")
        
        embeddings = []
        batch_size = self.config.get('batch_size', 32)
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_embeddings = self.model.predict(batch, verbose=0)
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        logger.info(f"Extracted {len(embeddings)} embeddings")
        
        return embeddings
    
    def evaluate_verification(self, embeddings: np.ndarray, labels: np.ndarray, 
                            label_names: List[str]) -> Dict:
        """
        Evaluate face verification performance
        
        Args:
            embeddings: Extracted embeddings
            labels: Ground truth labels
            label_names: Label names
            
        Returns:
            Dictionary with verification metrics
        """
        logger.info("Evaluating face verification")
        
        # Create positive and negative pairs
        positive_pairs = []
        negative_pairs = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                if labels[i] == labels[j]:
                    positive_pairs.append((i, j))
                else:
                    negative_pairs.append((i, j))
        
        # Sample pairs for evaluation
        max_pairs = min(1000, len(positive_pairs), len(negative_pairs))
        positive_pairs = positive_pairs[:max_pairs]
        negative_pairs = negative_pairs[:max_pairs]
        
        # Calculate similarities
        positive_similarities = []
        negative_similarities = []
        
        for i, j in positive_pairs:
            similarity = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            positive_similarities.append(similarity)
        
        for i, j in negative_pairs:
            similarity = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            negative_similarities.append(similarity)
        
        # Calculate metrics
        positive_similarities = np.array(positive_similarities)
        negative_similarities = np.array(negative_similarities)
        
        # Find optimal threshold
        all_similarities = np.concatenate([positive_similarities, negative_similarities])
        all_labels = np.concatenate([np.ones(len(positive_similarities)), 
                                   np.zeros(len(negative_similarities))])
        
        thresholds = np.linspace(0, 1, 100)
        best_threshold = 0
        best_accuracy = 0
        
        for threshold in thresholds:
            predictions = (all_similarities >= threshold).astype(int)
            accuracy = accuracy_score(all_labels, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        # Calculate final metrics
        final_predictions = (all_similarities >= best_threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, final_predictions, average='binary'
        )
        
        # Calculate EER (Equal Error Rate)
        eer = self.calculate_eer(positive_similarities, negative_similarities)
        
        verification_metrics = {
            'best_threshold': float(best_threshold),
            'accuracy': float(best_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'eer': float(eer),
            'positive_mean_similarity': float(np.mean(positive_similarities)),
            'negative_mean_similarity': float(np.mean(negative_similarities)),
            'positive_std_similarity': float(np.std(positive_similarities)),
            'negative_std_similarity': float(np.std(negative_similarities))
        }
        
        logger.info(f"Verification Accuracy: {best_accuracy:.4f}")
        logger.info(f"EER: {eer:.4f}")
        logger.info(f"Best Threshold: {best_threshold:.4f}")
        
        return verification_metrics
    
    def calculate_eer(self, positive_similarities: np.ndarray, 
                     negative_similarities: np.ndarray) -> float:
        """Calculate Equal Error Rate"""
        thresholds = np.linspace(0, 1, 1000)
        
        min_eer = 1.0
        for threshold in thresholds:
            far = np.mean(negative_similarities >= threshold)  # False Accept Rate
            frr = np.mean(positive_similarities < threshold)   # False Reject Rate
            
            eer = (far + frr) / 2
            if eer < min_eer:
                min_eer = eer
        
        return min_eer
    
    def evaluate_identification(self, embeddings: np.ndarray, labels: np.ndarray, 
                              label_names: List[str]) -> Dict:
        """
        Evaluate face identification performance
        
        Args:
            embeddings: Extracted embeddings
            labels: Ground truth labels
            label_names: Label names
            
        Returns:
            Dictionary with identification metrics
        """
        logger.info("Evaluating face identification")
        
        # Create database embeddings
        database_embeddings = {}
        for i, (embedding, label, name) in enumerate(zip(embeddings, labels, label_names)):
            if name not in database_embeddings:
                database_embeddings[name] = []
            database_embeddings[name].append(embedding)
        
        # Test identification
        correct_predictions = 0
        total_predictions = 0
        confidence_scores = []
        
        for i, (embedding, label, name) in enumerate(zip(embeddings, labels, label_names)):
            # Remove current embedding from database
            test_db = database_embeddings.copy()
            if name in test_db and len(test_db[name]) > 1:
                test_db[name] = [emb for j, emb in enumerate(test_db[name]) if j != i]
            elif name in test_db:
                continue  # Skip if only one embedding for this person
            
            # Predict identity
            result = self.comparator.compare_face(embedding, test_db)
            
            if result.identity == name:
                correct_predictions += 1
            
            total_predictions += 1
            confidence_scores.append(result.confidence)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        identification_metrics = {
            'accuracy': float(accuracy),
            'correct_predictions': int(correct_predictions),
            'total_predictions': int(total_predictions),
            'mean_confidence': float(np.mean(confidence_scores)),
            'std_confidence': float(np.std(confidence_scores))
        }
        
        logger.info(f"Identification Accuracy: {accuracy:.4f}")
        logger.info(f"Correct Predictions: {correct_predictions}/{total_predictions}")
        
        return identification_metrics
    
    def plot_similarity_distributions(self, positive_similarities: np.ndarray, 
                                    negative_similarities: np.ndarray):
        """Plot similarity distributions"""
        plt.figure(figsize=(10, 6))
        
        plt.hist(positive_similarities, bins=50, alpha=0.7, label='Positive Pairs', 
                color='green', density=True)
        plt.hist(negative_similarities, bins=50, alpha=0.7, label='Negative Pairs', 
                color='red', density=True)
        
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.title('Similarity Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.config['output_dir'], 'similarity_distributions.png'))
        plt.close()
        
        logger.info("Similarity distributions plotted")
    
    def plot_roc_curve(self, positive_similarities: np.ndarray, 
                      negative_similarities: np.ndarray):
        """Plot ROC curve"""
        thresholds = np.linspace(0, 1, 1000)
        tprs = []
        fprs = []
        
        for threshold in thresholds:
            tpr = np.mean(positive_similarities >= threshold)  # True Positive Rate
            fpr = np.mean(negative_similarities >= threshold)  # False Positive Rate
            tprs.append(tpr)
            fprs.append(fpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fprs, tprs, 'b-', linewidth=2)
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid(True, alpha=0.3)
        
        # Calculate AUC
        auc = np.trapz(tprs, fprs)
        plt.text(0.6, 0.2, f'AUC = {auc:.4f}', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        
        plt.savefig(os.path.join(self.config['output_dir'], 'roc_curve.png'))
        plt.close()
        
        logger.info(f"ROC curve plotted (AUC = {auc:.4f})")
        return auc
    
    def save_results(self, verification_metrics: Dict, identification_metrics: Dict, 
                    auc: float):
        """Save evaluation results"""
        results = {
            'evaluation_date': datetime.now().isoformat(),
            'model_path': self.config['model_path'],
            'test_data_path': self.config['test_data_path'],
            'verification_metrics': verification_metrics,
            'identification_metrics': identification_metrics,
            'auc': float(auc)
        }
        
        results_path = os.path.join(self.config['output_dir'], 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
    
    def run_evaluation(self):
        """Run complete evaluation"""
        logger.info("Starting model evaluation")
        
        try:
            # Load model
            self.load_model(self.config['model_path'])
            
            # Load test data
            images, labels, label_names = self.load_test_data(self.config['test_data_path'])
            
            # Extract embeddings
            embeddings = self.extract_embeddings(images)
            
            # Evaluate verification
            verification_metrics = self.evaluate_verification(embeddings, labels, label_names)
            
            # Evaluate identification
            identification_metrics = self.evaluate_identification(embeddings, labels, label_names)
            
            # Create positive and negative pairs for plotting
            positive_pairs = []
            negative_pairs = []
            
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    if labels[i] == labels[j]:
                        positive_pairs.append((i, j))
                    else:
                        negative_pairs.append((i, j))
            
            # Sample pairs for plotting
            max_pairs = min(1000, len(positive_pairs), len(negative_pairs))
            positive_pairs = positive_pairs[:max_pairs]
            negative_pairs = negative_pairs[:max_pairs]
            
            # Calculate similarities for plotting
            positive_similarities = []
            negative_similarities = []
            
            for i, j in positive_pairs:
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                positive_similarities.append(similarity)
            
            for i, j in negative_pairs:
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                negative_similarities.append(similarity)
            
            positive_similarities = np.array(positive_similarities)
            negative_similarities = np.array(negative_similarities)
            
            # Plot results
            self.plot_similarity_distributions(positive_similarities, negative_similarities)
            auc = self.plot_roc_curve(positive_similarities, negative_similarities)
            
            # Save results
            self.save_results(verification_metrics, identification_metrics, auc)
            
            logger.info("Evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Evaluate FaceNet model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Evaluation configuration
    config = {
        'model_path': args.model_path,
        'test_data_path': args.test_data,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size
    }
    
    # Create evaluator
    evaluator = FaceNetEvaluator(config)
    
    try:
        # Run evaluation
        evaluator.run_evaluation()
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
