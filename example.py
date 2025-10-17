"""
Example usage of the Face Recognition System
"""

import argparse
import os
from face_recognition_system import FaceRecognitionSystem


def main():
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--mode', choices=['detect', 'recognize'], default='detect',
                       help='Mode: detect (just detect faces) or recognize (recognize known faces)')
    parser.add_argument('--input', required=True, help='Path to input image')
    parser.add_argument('--output', default='output.jpg', help='Path to output image')
    parser.add_argument('--known-faces', default='known_faces', 
                       help='Directory containing known face images')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return
    
    # Initialize face recognition system
    frs = FaceRecognitionSystem(known_faces_dir=args.known_faces)
    
    if args.mode == 'detect':
        print("Mode: Face Detection")
        face_locations = frs.detect_faces(args.input)
        print(f"Detected {len(face_locations)} face(s)")
        frs.draw_faces(args.input, args.output, recognize=False)
    else:
        print("Mode: Face Recognition")
        results = frs.recognize_faces(args.input)
        print(f"Recognized {len(results)} face(s):")
        for result in results:
            print(f"  - {result['name']} (confidence: {result['confidence']:.2f})")
        frs.draw_faces(args.input, args.output, recognize=True)
    
    print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()
