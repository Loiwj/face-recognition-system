"""
Fine-tuning script cho FaceNet model
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import cv2
import json
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.face_detection import FaceDetectorFactory
from modules.preprocessing import FacePreprocessor, create_preprocessing_config, PreprocessingMode
from modules.facenet_embedding import FaceNetFactory, FaceNetConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class FaceNetFineTuner:
    """Class để fine-tune FaceNet model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.history = None
        
        # Initialize components
        self.face_detector = FaceDetectorFactory.create_detector('mtcnn')
        self.preprocessor = FacePreprocessor(create_preprocessing_config(PreprocessingMode.TRAINING))
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        logger.info("FaceNetFineTuner initialized")
    
    def load_pretrained_model(self, model_path: str) -> keras.Model:
        """Load pre-trained model"""
        logger.info(f"Loading pre-trained model from {model_path}")
        
        try:
            model = keras.models.load_model(model_path, compile=False)
            logger.info("Pre-trained model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading pre-trained model: {e}")
            raise
    
    def load_custom_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load custom dataset for fine-tuning
        
        Args:
            data_path: Đường dẫn đến custom dataset
            
        Returns:
            Tuple (images, labels)
        """
        logger.info(f"Loading custom dataset from {data_path}")
        
        images = []
        labels = []
        label_to_id = {}
        current_id = 0
        
        # Walk through dataset directory
        for root, dirs, files in os.walk(data_path):
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
                                    image, faces, PreprocessingMode.TRAINING
                                )
                                
                                if processed_faces:
                                    images.append(processed_faces[0])
                                    labels.append(person_id)
                                    
                        except Exception as e:
                            logger.warning(f"Error processing {image_path}: {e}")
                            continue
        
        if not images:
            raise ValueError("No valid images found in custom dataset")
        
        images = np.array(images)
        labels = np.array(labels)
        
        logger.info(f"Loaded {len(images)} images for {len(label_to_id)} identities")
        logger.info(f"Image shape: {images.shape}")
        
        # Save label mapping
        id_to_label = {v: k for k, v in label_to_id.items()}
        with open(os.path.join(self.config['output_dir'], 'custom_label_mapping.json'), 'w') as f:
            json.dump(id_to_label, f, indent=2)
        
        return images, labels
    
    def prepare_model_for_fine_tuning(self, model: keras.Model) -> keras.Model:
        """Prepare model for fine-tuning"""
        logger.info("Preparing model for fine-tuning")
        
        # Unfreeze some layers for fine-tuning
        for layer in model.layers[-10:]:  # Unfreeze last 10 layers
            layer.trainable = True
        
        # Add new classification head if needed
        if self.config.get('add_classification_head', False):
            # Add classification head
            x = model.output
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(len(np.unique(self.config.get('num_classes', 1000))), 
                           activation='softmax')(x)
            
            model = keras.Model(inputs=model.input, outputs=x)
        
        # Compile with lower learning rate for fine-tuning
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['fine_tune_lr']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model prepared for fine-tuning")
        return model
    
    def fine_tune(self, images: np.ndarray, labels: np.ndarray):
        """Fine-tune the model"""
        logger.info("Starting fine-tuning")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        
        # Prepare model
        self.model = self.prepare_model_for_fine_tuning(self.model)
        
        # Callbacks
        callbacks_list = [
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config['output_dir'], 'fine_tuned_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            callbacks.TensorBoard(
                log_dir=os.path.join(self.config['output_dir'], 'fine_tune_logs'),
                histogram_freq=1
            )
        ]
        
        # Fine-tuning
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['fine_tune_epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        logger.info("Fine-tuning completed")
    
    def evaluate_model(self, test_images: np.ndarray, test_labels: np.ndarray):
        """Evaluate fine-tuned model"""
        logger.info("Evaluating model")
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(test_images, test_labels, verbose=0)
        
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Save evaluation results
        eval_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'evaluation_date': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.config['output_dir'], 'evaluation_results.json'), 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        return eval_results
    
    def save_model(self, model_path: str):
        """Save fine-tuned model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(model_path)
        logger.info(f"Fine-tuned model saved to {model_path}")
        
        # Save training history
        history_path = os.path.join(self.config['output_dir'], 'fine_tune_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history.history, f, indent=2)
        
        # Plot training history
        self.plot_history()
    
    def plot_history(self):
        """Plot fine-tuning history"""
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Fine-tuning Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Fine-tuning Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'fine_tune_history.png'))
        plt.close()
        
        logger.info("Fine-tuning history plotted")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune FaceNet model')
    parser.add_argument('--pretrained_model', type=str, required=True,
                       help='Path to pre-trained model')
    parser.add_argument('--custom_data', type=str, required=True,
                       help='Path to custom dataset')
    parser.add_argument('--output_dir', type=str, default='./models/fine_tuned',
                       help='Output directory for fine-tuned model')
    parser.add_argument('--fine_tune_epochs', type=int, default=20,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for fine-tuning')
    parser.add_argument('--fine_tune_lr', type=float, default=0.0001,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--add_classification_head', action='store_true',
                       help='Add classification head for fine-tuning')
    parser.add_argument('--model_name', type=str, default='fine_tuned_facenet',
                       help='Name for the fine-tuned model')
    
    args = parser.parse_args()
    
    # Fine-tuning configuration
    config = {
        'pretrained_model': args.pretrained_model,
        'custom_data': args.custom_data,
        'output_dir': args.output_dir,
        'fine_tune_epochs': args.fine_tune_epochs,
        'batch_size': args.batch_size,
        'fine_tune_lr': args.fine_tune_lr,
        'add_classification_head': args.add_classification_head,
        'model_name': args.model_name
    }
    
    # Create fine-tuner
    fine_tuner = FaceNetFineTuner(config)
    
    try:
        # Load pre-trained model
        model = fine_tuner.load_pretrained_model(args.pretrained_model)
        fine_tuner.model = model
        
        # Load custom data
        images, labels = fine_tuner.load_custom_data(args.custom_data)
        
        # Fine-tune model
        fine_tuner.fine_tune(images, labels)
        
        # Evaluate model
        eval_results = fine_tuner.evaluate_model(images, labels)
        
        # Save model
        model_path = os.path.join(args.output_dir, f'{args.model_name}.h5')
        fine_tuner.save_model(model_path)
        
        logger.info(f"Fine-tuning completed successfully. Model saved to {model_path}")
        logger.info(f"Final accuracy: {eval_results['test_accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
