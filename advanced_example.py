"""
Advanced examples for the Face Recognition System
Demonstrates various use cases and features
"""

import os
import sys
from face_recognition_system import FaceRecognitionSystem


def example_1_basic_detection():
    """Example 1: Basic face detection"""
    print("\n=== Example 1: Basic Face Detection ===")
    
    frs = FaceRecognitionSystem()
    
    # This example would detect faces without recognition
    # frs.detect_faces("input_image.jpg")
    print("To use: frs.detect_faces('input_image.jpg')")
    print("Returns: List of (x, y, width, height) tuples for each detected face")


def example_2_face_recognition():
    """Example 2: Face recognition with known faces"""
    print("\n=== Example 2: Face Recognition ===")
    
    # Initialize with known faces directory
    frs = FaceRecognitionSystem(known_faces_dir="known_faces")
    
    # This would recognize faces in the image
    # results = frs.recognize_faces("test_image.jpg")
    # for result in results:
    #     print(f"Found: {result['name']} (confidence: {result['confidence']:.2f})")
    
    print("To use: results = frs.recognize_faces('test_image.jpg')")
    print("Returns: List of dicts with 'name', 'location', and 'confidence'")


def example_3_batch_processing():
    """Example 3: Process multiple images"""
    print("\n=== Example 3: Batch Processing ===")
    
    frs = FaceRecognitionSystem(known_faces_dir="known_faces")
    
    # Process all images in a directory
    input_dir = "input_images"
    output_dir = "output_images"
    
    print(f"""
To process multiple images:
    
    import os
    from face_recognition_system import FaceRecognitionSystem
    
    frs = FaceRecognitionSystem(known_faces_dir="known_faces")
    
    # Create output directory
    os.makedirs("{output_dir}", exist_ok=True)
    
    # Process each image
    for filename in os.listdir("{input_dir}"):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join("{input_dir}", filename)
            output_path = os.path.join("{output_dir}", filename)
            
            print(f"Processing {{filename}}...")
            frs.draw_faces(input_path, output_path, recognize=True)
    """)


def example_4_statistics():
    """Example 4: Gather statistics from images"""
    print("\n=== Example 4: Gathering Statistics ===")
    
    print("""
To gather statistics about faces in images:
    
    from face_recognition_system import FaceRecognitionSystem
    import os
    
    frs = FaceRecognitionSystem(known_faces_dir="known_faces")
    
    total_faces = 0
    recognized_faces = 0
    person_counts = {}
    
    for filename in os.listdir("images"):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join("images", filename)
            results = frs.recognize_faces(filepath)
            
            total_faces += len(results)
            
            for result in results:
                if result['name'] != "Unknown":
                    recognized_faces += 1
                    person_counts[result['name']] = person_counts.get(result['name'], 0) + 1
    
    print(f"Total faces detected: {total_faces}")
    print(f"Recognized faces: {recognized_faces}")
    print("Person counts:", person_counts)
    """)


def example_5_custom_settings():
    """Example 5: Using custom settings"""
    print("\n=== Example 5: Custom Settings ===")
    
    print("""
To customize face detection parameters, modify the face_recognition_system.py:
    
    # In the detect_faces method, you can adjust:
    faces = self.face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,    # Adjust for scale (1.1 is default, try 1.05-1.3)
        minNeighbors=5,     # Adjust for sensitivity (3-6 typical range)
        minSize=(30, 30)    # Minimum face size to detect
    )
    
    # Lower scaleFactor = more thorough but slower
    # Higher minNeighbors = fewer false positives but might miss faces
    # Larger minSize = ignore small/distant faces
    """)


def example_6_real_time_webcam():
    """Example 6: Real-time webcam face recognition"""
    print("\n=== Example 6: Real-time Webcam Recognition ===")
    
    print("""
For real-time webcam face recognition, add this to a new file:
    
    import cv2
    from face_recognition_system import FaceRecognitionSystem
    
    frs = FaceRecognitionSystem(known_faces_dir="known_faces")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = frs.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display
        cv2.imshow('Face Detection', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    """)


def main():
    """Run all examples"""
    print("=" * 60)
    print("Face Recognition System - Advanced Examples")
    print("=" * 60)
    
    examples = [
        example_1_basic_detection,
        example_2_face_recognition,
        example_3_batch_processing,
        example_4_statistics,
        example_5_custom_settings,
        example_6_real_time_webcam
    ]
    
    for example in examples:
        example()
    
    print("\n" + "=" * 60)
    print("For more information, see README.md and QUICKSTART.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
