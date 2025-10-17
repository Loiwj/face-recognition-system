"""
Tests for Face Recognition System
"""

import unittest
import os
import tempfile
import shutil
from face_recognition_system import FaceRecognitionSystem


class TestFaceRecognitionSystem(unittest.TestCase):
    """Test cases for the FaceRecognitionSystem class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.known_faces_dir = os.path.join(self.test_dir, "known_faces")
        os.makedirs(self.known_faces_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test system initialization"""
        frs = FaceRecognitionSystem(known_faces_dir=self.known_faces_dir)
        self.assertIsNotNone(frs)
        self.assertEqual(len(frs.known_face_names), 0)
        self.assertFalse(frs.is_trained)
    
    def test_initialization_with_nonexistent_directory(self):
        """Test initialization with non-existent directory"""
        frs = FaceRecognitionSystem(known_faces_dir="nonexistent_dir")
        self.assertIsNotNone(frs)
        self.assertFalse(frs.is_trained)
    
    def test_detect_faces_with_invalid_path(self):
        """Test face detection with invalid image path"""
        frs = FaceRecognitionSystem(known_faces_dir=self.known_faces_dir)
        result = frs.detect_faces("nonexistent_image.jpg")
        self.assertEqual(result, [])
    
    def test_recognize_faces_without_known_faces(self):
        """Test face recognition without any known faces"""
        frs = FaceRecognitionSystem(known_faces_dir=self.known_faces_dir)
        result = frs.recognize_faces("nonexistent_image.jpg")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
