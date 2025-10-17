"""
Face Recognition System
A simple face detection and recognition system using OpenCV.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FaceRecognitionSystem:
    """
    A class to handle face detection and recognition using OpenCV.
    """
    
    def __init__(self, known_faces_dir: str = "known_faces"):
        """
        Initialize the face recognition system.
        
        Args:
            known_faces_dir: Directory containing known face images
        """
        self.known_faces_dir = known_faces_dir
        self.known_face_descriptors = []
        self.known_face_names = []
        
        # Load face detection cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            logger.error("Failed to load face cascade classifier")
        
        # Initialize face recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.is_trained = False
        
        if os.path.exists(known_faces_dir):
            self.load_known_faces()
        else:
            logger.warning(f"Known faces directory '{known_faces_dir}' does not exist.")
    
    def load_known_faces(self):
        """
        Load all known faces from the known_faces directory.
        Each image file should be named with the person's name (e.g., john_doe.jpg).
        """
        logger.info(f"Loading known faces from {self.known_faces_dir}...")
        
        if not os.path.exists(self.known_faces_dir):
            logger.error(f"Directory {self.known_faces_dir} does not exist.")
            return
        
        faces = []
        labels = []
        
        for idx, filename in enumerate(sorted(os.listdir(self.known_faces_dir))):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(self.known_faces_dir, filename)
                try:
                    # Load image in grayscale
                    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    
                    if image is None:
                        logger.warning(f"Could not load {filename}")
                        continue
                    
                    # Detect faces in the image
                    detected_faces = self.face_cascade.detectMultiScale(
                        image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    
                    if len(detected_faces) > 0:
                        # Use the first detected face
                        x, y, w, h = detected_faces[0]
                        face_roi = image[y:y+h, x:x+w]
                        
                        # Resize to standard size
                        face_roi = cv2.resize(face_roi, (100, 100))
                        
                        faces.append(face_roi)
                        labels.append(idx)
                        
                        # Use filename (without extension) as the person's name
                        name = os.path.splitext(filename)[0]
                        self.known_face_names.append(name)
                        logger.info(f"Loaded face: {name}")
                    else:
                        logger.warning(f"No face found in {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {str(e)}")
        
        # Train the recognizer if we have faces
        if len(faces) > 0:
            self.face_recognizer.train(faces, np.array(labels))
            self.is_trained = True
            logger.info(f"Trained recognizer with {len(self.known_face_names)} known faces.")
        else:
            logger.warning("No faces loaded for training.")
    
    def detect_faces(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of face locations (x, y, width, height)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return []
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            logger.info(f"Found {len(faces)} face(s) in {image_path}")
            return [tuple(face) for face in faces]
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {str(e)}")
            return []
    
    def recognize_faces(self, image_path: str) -> List[Dict[str, any]]:
        """
        Recognize faces in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries containing face information (name, location, confidence)
        """
        if not self.is_trained:
            logger.warning("No known faces loaded. Please add faces to the known_faces directory.")
            return []
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return []
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            face_locations = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            results = []
            
            for (x, y, w, h) in face_locations:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                
                # Predict
                label, confidence = self.face_recognizer.predict(face_roi)
                
                # Convert confidence (lower is better) to percentage
                # LBPH returns values typically between 0-150+
                # We'll invert and normalize it
                confidence_percent = max(0, 1.0 - (confidence / 100.0))
                
                name = "Unknown"
                # If confidence is good enough (distance < 70 typically), use the predicted name
                if confidence < 70 and label < len(self.known_face_names):
                    name = self.known_face_names[label]
                
                results.append({
                    'name': name,
                    'location': (x, y, w, h),
                    'confidence': confidence_percent
                })
                
                logger.info(f"Recognized: {name} (confidence: {confidence_percent:.2f}, distance: {confidence:.2f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error recognizing faces in {image_path}: {str(e)}")
            return []
    
    def draw_faces(self, image_path: str, output_path: str, recognize: bool = True):
        """
        Draw rectangles around detected faces and save the result.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image
            recognize: If True, perform recognition; if False, just detection
        """
        try:
            # Load image with OpenCV
            image = cv2.imread(image_path)
            
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return
            
            if recognize and self.is_trained:
                # Recognize faces
                face_results = self.recognize_faces(image_path)
                
                for result in face_results:
                    x, y, w, h = result['location']
                    name = result['name']
                    confidence = result['confidence']
                    
                    # Draw rectangle
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw label
                    label = f"{name} ({confidence:.2f})" if name != "Unknown" else name
                    cv2.rectangle(image, (x, y-35), (x+w, y), color, cv2.FILLED)
                    cv2.putText(image, label, (x + 6, y - 6),
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            else:
                # Just detect faces
                face_locations = self.detect_faces(image_path)
                
                for (x, y, w, h) in face_locations:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Save image
            cv2.imwrite(output_path, image)
            logger.info(f"Saved result to {output_path}")
            
        except Exception as e:
            logger.error(f"Error drawing faces: {str(e)}")
