"""
Training script cho FaceNet model
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


class FaceNetTrainer:
    """Class để train FaceNet model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.history = None
        
        # Initialize components
        self.face_detector = FaceDetectorFactory.create_detector('mtcnn')
        self.preprocessor = FacePreprocessor(create_preprocessing_config(PreprocessingMode.TRAINING))
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        logger.info("FaceNetTrainer initialized")
    
    def load_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load và preprocess dataset
        
        Args:
            dataset_path: Đường dẫn đến dataset
            
        Returns:
            Tuple (images, labels)
        """
        logger.info(f"Loading dataset from {dataset_path}")
        
        images = []
        labels = []
        label_to_id = {}
        current_id = 0
        
        # Walk through dataset directory
        for root, dirs, files in os.walk(dataset_path):
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
            raise ValueError("No valid images found in dataset")
        
        images = np.array(images)
        labels = np.array(labels)
        
        logger.info(f"Loaded {len(images)} images for {len(label_to_id)} identities")
        logger.info(f"Image shape: {images.shape}")
        
        # Save label mapping
        id_to_label = {v: k for k, v in label_to_id.items()}
        with open(os.path.join(self.config['output_dir'], 'label_mapping.json'), 'w') as f:
            json.dump(id_to_label, f, indent=2)
        
        return images, labels
    
    def create_model(self) -> keras.Model:
        """Tạo FaceNet model"""
        logger.info("Creating FaceNet model")
        
        # Input layer
        input_layer = layers.Input(shape=(160, 160, 3), name='input_image')
        
        # Base model (Inception ResNet v1)
        base_model = keras.applications.InceptionResNetV2(
            weights='imagenet',
            include_top=False,
            input_tensor=input_layer
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Custom layers
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu', name='fc1')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, name='fc2')(x)
        
        # L2 normalization
        x = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1), name='l2_norm')(x)
        
        # Create model
        model = keras.Model(inputs=input_layer, outputs=x, name='FaceNet')
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=self.triplet_loss,
            metrics=['accuracy']
        )
        
        logger.info(f"Model created with {model.count_params()} parameters")
        return model
    
    def triplet_loss(self, y_true, y_pred, margin=0.2):
        """Triplet loss function"""
        anchor, positive, negative = tf.split(y_pred, 3, axis=0)
        
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
        return tf.reduce_mean(loss)
    
    def generate_triplets(self, images: np.ndarray, labels: np.ndarray, 
                         batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate triplets for training
        
        Args:
            images: Training images
            labels: Training labels
            batch_size: Batch size
            
        Returns:
            Tuple (triplet_images, triplet_labels)
        """
        # Group images by label
        label_to_indices = {}
        for i, label in enumerate(labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(i)
        
        # Generate triplets
        triplet_images = []
        triplet_labels = []
        
        for _ in range(batch_size):
            # Random anchor
            anchor_label = np.random.choice(list(label_to_indices.keys()))
            anchor_idx = np.random.choice(label_to_indices[anchor_label])
            
            # Positive (same person)
            positive_indices = [i for i in label_to_indices[anchor_label] if i != anchor_idx]
            if positive_indices:
                positive_idx = np.random.choice(positive_indices)
            else:
                positive_idx = anchor_idx
            
            # Negative (different person)
            negative_labels = [l for l in label_to_indices.keys() if l != anchor_label]
            negative_label = np.random.choice(negative_labels)
            negative_idx = np.random.choice(label_to_indices[negative_label])
            
            # Add triplet
            triplet_images.extend([
                images[anchor_idx],
                images[positive_idx],
                images[negative_idx]
            ])
            triplet_labels.extend([anchor_label, anchor_label, negative_label])
        
        return np.array(triplet_images), np.array(triplet_labels)
    
    def train(self, images: np.ndarray, labels: np.ndarray):
        """Train the model"""
        logger.info("Starting training")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        
        # Create model
        self.model = self.create_model()
        
        # Callbacks
        callbacks_list = [
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config['output_dir'], 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.TensorBoard(
                log_dir=os.path.join(self.config['output_dir'], 'logs'),
                histogram_freq=1
            )
        ]
        
        # Training loop
        for epoch in range(self.config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            
            # Generate triplets for training
            X_triplet, y_triplet = self.generate_triplets(
                X_train, y_train, self.config['batch_size']
            )
            
            # Train
            history = self.model.fit(
                X_triplet, y_triplet,
                batch_size=self.config['batch_size'],
                epochs=1,
                validation_data=(X_val, y_val),
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Save epoch history
            if self.history is None:
                self.history = history.history
            else:
                for key in history.history:
                    self.history[key].extend(history.history[key])
        
        logger.info("Training completed")
    
    def save_model(self, model_path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save training history
        history_path = os.path.join(self.config['output_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Plot training history
        self.plot_history()
    
    def plot_history(self):
        """Plot training history"""
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history['loss'], label='Training Loss')
        if 'val_loss' in self.history:
            ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        if 'accuracy' in self.history:
            ax2.plot(self.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in self.history:
                ax2.plot(self.history['val_accuracy'], label='Validation Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'training_history.png'))
        plt.close()
        
        logger.info("Training history plotted")


def main():
    parser = argparse.ArgumentParser(description='Train FaceNet model')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./models/trained',
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model_name', type=str, default='custom_facenet',
                       help='Name for the trained model')
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'dataset_path': args.dataset_path,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'model_name': args.model_name
    }
    
    # Create trainer
    trainer = FaceNetTrainer(config)
    
    try:
        # Load dataset
        images, labels = trainer.load_dataset(args.dataset_path)
        
        # Train model
        trainer.train(images, labels)
        
        # Save model
        model_path = os.path.join(args.output_dir, f'{args.model_name}.h5')
        trainer.save_model(model_path)
        
        logger.info(f"Training completed successfully. Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
