# Quick Start Guide

This guide will help you get started with the Face Recognition System in just a few minutes.

## Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/Loiwj/face-recognition-system.git
cd face-recognition-system

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Prepare Known Faces

Create a directory for known faces and add images:

```bash
mkdir known_faces
# Add images of people you want to recognize
# Name them like: john_doe.jpg, jane_smith.jpg, etc.
```

**Important**: Each image should:
- Contain only one face
- Be clear and well-lit
- Be named after the person (without spaces recommended)

## Step 3: Run Face Detection

Detect faces in an image without recognition:

```bash
python example.py --mode detect --input your_photo.jpg --output result.jpg
```

## Step 4: Run Face Recognition

Once you have added known faces, you can recognize them:

```bash
python example.py --mode recognize --input your_photo.jpg --output result.jpg
```

## Example Usage in Code

```python
from face_recognition_system import FaceRecognitionSystem

# Initialize the system
frs = FaceRecognitionSystem(known_faces_dir="known_faces")

# Detect faces
faces = frs.detect_faces("photo.jpg")
print(f"Found {len(faces)} faces")

# Recognize faces (if known faces are loaded)
results = frs.recognize_faces("photo.jpg")
for result in results:
    print(f"{result['name']}: {result['confidence']:.2%}")

# Draw faces on image and save
frs.draw_faces("photo.jpg", "output.jpg", recognize=True)
```

## Tips

1. **Better Recognition**: Add multiple photos of each person for better accuracy
2. **Image Quality**: Use clear, front-facing photos with good lighting
3. **Confidence Scores**: Higher confidence (closer to 1.0) means better match
4. **Batch Processing**: You can process multiple images in a loop

## Common Issues

**No faces detected?**
- Ensure faces are clearly visible and well-lit
- Try different scaleFactor values (requires code modification)
- Check if the image is corrupted

**Low recognition accuracy?**
- Add more training images per person
- Ensure training images are high quality
- Make sure faces are front-facing in training images

## Next Steps

- Check out the [README.md](README.md) for detailed documentation
- Run the unit tests: `python -m unittest test_face_recognition.py`
- Explore the code in `face_recognition_system.py` to understand how it works
