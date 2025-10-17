# Face Recognition System

A simple yet powerful face detection and recognition system built with Python and OpenCV.

## Features

- **Face Detection**: Detect faces in images
- **Face Recognition**: Recognize known faces with confidence scores
- **Easy to Use**: Simple command-line interface
- **Extensible**: Easy to add new faces to the system

## Requirements

- Python 3.7+
- OpenCV (with contrib modules)
- NumPy
- Pillow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Loiwj/face-recognition-system.git
cd face-recognition-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: The system uses OpenCV's built-in face detection and recognition algorithms, which are easy to install and don't require additional dependencies.

## Usage

### Face Detection

Detect faces in an image without recognition:

```bash
python example.py --mode detect --input path/to/image.jpg --output result.jpg
```

### Face Recognition

Recognize known faces in an image:

1. First, create a `known_faces` directory and add images of people you want to recognize:
```bash
mkdir known_faces
# Add images named after the person (e.g., john_doe.jpg, jane_smith.jpg)
```

2. Run face recognition:
```bash
python example.py --mode recognize --input path/to/image.jpg --output result.jpg
```

### Command-line Arguments

- `--mode`: Choose between `detect` (just detect faces) or `recognize` (recognize known faces)
- `--input`: Path to the input image (required)
- `--output`: Path to save the output image (default: `output.jpg`)
- `--known-faces`: Directory containing known face images (default: `known_faces`)

## Project Structure

```
face-recognition-system/
├── face_recognition_system.py  # Main face recognition module
├── example.py                  # Example usage script
├── requirements.txt            # Python dependencies
├── known_faces/               # Directory for known face images
└── README.md                  # This file
```

## How It Works

1. **Loading Known Faces**: The system loads images from the `known_faces` directory and trains a recognizer.
2. **Face Detection**: Uses OpenCV's Haar Cascade classifier to detect faces.
3. **Face Recognition**: Uses the LBPH (Local Binary Patterns Histograms) algorithm for face recognition.
4. **Training**: Trains the recognizer with the known faces for better accuracy.
5. **Visualization**: Draws bounding boxes and labels on detected faces.

## Example

```python
from face_recognition_system import FaceRecognitionSystem

# Initialize the system
frs = FaceRecognitionSystem(known_faces_dir="known_faces")

# Detect faces
face_locations = frs.detect_faces("test_image.jpg")
print(f"Found {len(face_locations)} faces")

# Recognize faces
results = frs.recognize_faces("test_image.jpg")
for result in results:
    print(f"Name: {result['name']}, Confidence: {result['confidence']:.2f}")

# Draw and save result
frs.draw_faces("test_image.jpg", "output.jpg", recognize=True)
```

## Performance Tips

- Use smaller images for faster processing
- The system works best with clear, front-facing photos
- Good lighting improves recognition accuracy
- Multiple images per person can improve recognition

## Troubleshooting

**Issue**: No faces detected
- **Solution**: Ensure the image has sufficient resolution and faces are clearly visible

**Issue**: Low recognition accuracy
- **Solution**: Add more reference images of the person, ensure good lighting and image quality

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with OpenCV and its computer vision algorithms
- Uses Haar Cascade for face detection and LBPH for face recognition