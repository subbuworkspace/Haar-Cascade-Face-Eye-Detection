# 👤 Haar Cascade Face & Eye Detection using OpenCV

A beginner-friendly **Computer Vision** project using **OpenCV and Haar Cascade Classifiers** to detect human faces and eyes from an image.

This project demonstrates how traditional Computer Vision techniques can be used for object detection without requiring a Deep Learning model.

---

## 📌 Project Overview

The project uses OpenCV's **Haar Cascade Classifier** to:

* Detect human faces
* Detect eyes inside detected faces
* Draw bounding boxes around faces
* Draw bounding boxes around eyes
* Display the processed image

### Detection Flow

```text
Input Image
     ↓
Convert BGR → Grayscale
     ↓
Haar Cascade Face Detection
     ↓
Detect Face
     ↓
Extract Face ROI
     ↓
Haar Cascade Eye Detection
     ↓
Draw Bounding Boxes
     ↓
Display Result
```

---

## 🧠 What is Haar Cascade?

**Haar Cascade** is a traditional machine-learning-based object detection method introduced by Paul Viola and Michael Jones.

It uses a cascade of classifiers to identify objects such as:

* Face
* Eyes
* Nose
* Full body
* Smile

OpenCV provides pre-trained Haar Cascade XML files that can be used directly for detection.

---

## 🔍 How Haar Cascade Works

The basic process is:

```text
Image
  ↓
Grayscale Conversion
  ↓
Haar-like Features
  ↓
Integral Image
  ↓
AdaBoost
  ↓
Cascade of Classifiers
  ↓
Object Detection
```

### 1. Haar-like Features

Haar features compare the difference between light and dark regions.

For example:

```text
White Region | Black Region
████████████ | ░░░░░░░░░░░
```

For a face, certain regions such as the eye area tend to be darker than the forehead or cheeks.

---

### 2. Integral Image

Integral images make it faster to calculate the sum of pixels inside rectangular regions.

Instead of repeatedly calculating pixel sums, Haar Cascade can calculate them efficiently.

---

### 3. AdaBoost

AdaBoost selects the most useful features from many possible Haar features.

The selected features are combined to create a strong classifier.

---

### 4. Cascade Classifier

The detector uses multiple stages.

```text
Stage 1 → Quick rejection
Stage 2 → More checking
Stage 3 → More checking
Stage 4 → Face detected
```

Most non-face regions are rejected early, making the detection relatively fast.

---

# 🛠️ Technologies Used

* Python
* OpenCV
* NumPy
* Haar Cascade Classifier
* Git
* GitHub

---

# 📦 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Haar-Cascade-Face-Eye-Detection.git
```

Move into the project:

```bash
cd Haar-Cascade-Face-Eye-Detection
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate it in Git Bash:

```bash
source .venv/Scripts/activate
```

Or in Command Prompt:

```cmd
.venv\Scripts\activate
```

---

## 3. Install Required Packages

```bash
python -m pip install opencv-python==4.10.0.84
python -m pip install numpy
```

Or install everything from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

---

# 📁 Project Structure

```text
Haar-Cascade-Face-Eye-Detection/
│
├── Haarcascades/
│   ├── haarcascade_frontalface_default.xml
│   └── haarcascade_eye.xml
│
├── faceinput.jpg
│
├── face_eye_detection.py
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

---

# 💻 Python Code

```python
import cv2

# ---------------------------------------
# 1. Load Haar Cascade Classifiers
# ---------------------------------------

face_path = r"Haarcascades\haarcascade_frontalface_default.xml"
eye_path = r"Haarcascades\haarcascade_eye.xml"

face_classifier = cv2.CascadeClassifier(face_path)
eye_classifier = cv2.CascadeClassifier(eye_path)

# Check classifiers
if face_classifier.empty():
    print("Error: Face Haar Cascade not loaded!")
    exit()

if eye_classifier.empty():
    print("Error: Eye Haar Cascade not loaded!")
    exit()

print("Haar Cascade files loaded successfully!")


# ---------------------------------------
# 2. Load Image
# ---------------------------------------

image_path = "faceinput.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found!")
    exit()

print("Image loaded successfully!")


# ---------------------------------------
# 3. Convert Image to Grayscale
# ---------------------------------------

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------
# 4. Detect Faces
# ---------------------------------------

faces = face_classifier.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5
)

print("Faces detected:", len(faces))


# ---------------------------------------
# 5. Detect Eyes
# ---------------------------------------

for (x, y, w, h) in faces:

    # Draw rectangle around face
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (127, 0, 255),
        2
    )

    # Face Region of Interest
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # Detect eyes
    eyes = eye_classifier.detectMultiScale(roi_gray)

    print("Eyes detected:", len(eyes))

    for (ex, ey, ew, eh) in eyes:

        # Draw rectangle around eyes
        cv2.rectangle(
            roi_color,
            (ex, ey),
            (ex + ew, ey + eh),
            (255, 255, 0),
            2
        )


# ---------------------------------------
# 6. Display Result
# ---------------------------------------

cv2.imshow("Face and Eye Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

# ⚙️ Important Parameters

The main detection function is:

```python
faces = face_classifier.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5
)
```

### `scaleFactor`

Controls how much the image size is reduced at each scale.

Example:

```python
scaleFactor=1.1
```

More detailed detection but potentially slower.

```python
scaleFactor=1.3
```

Faster detection with fewer scales.

---

### `minNeighbors`

Controls how many neighboring detections are required before considering an object a valid detection.

Example:

```python
minNeighbors=5
```

Increasing it can reduce false positives.

```python
minNeighbors=3
```

May detect more objects but can produce more false positives.

---

# 🎯 Region of Interest (ROI)

After detecting a face:

```python
roi_gray = gray[y:y+h, x:x+w]
```

we only search for eyes inside that face.

This is called a **Region of Interest (ROI)**.

Instead of searching the entire image:

```text
Entire Image
┌─────────────────────────┐
│                         │
│       FACE              │
│      👁  👁             │
│                         │
│                         │
└─────────────────────────┘
```

we search only:

```text
Face ROI
┌─────────────┐
│   👁   👁   │
│             │
│             │
└─────────────┘
```

This makes eye detection more focused.

---

# 🧪 Example

Input:

```text
faceinput.jpg
```

Output:

```text
Face
┌─────────────────────┐
│                     │
│     ┌───┐   ┌───┐   │
│     │Eye│   │Eye│   │
│     └───┘   └───┘   │
│                     │
└─────────────────────┘
```

The program draws:

* Purple rectangle → Face
* Yellow rectangle → Eyes

---

# ❓ Why Convert the Image to Grayscale?

We use:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

Haar Cascade detection generally works on grayscale intensity information rather than needing full RGB/BGR color information.

Advantages:

* Fewer channels
* Less computational work
* Simpler image representation
* Faster processing

---

# 📚 Important OpenCV Functions

### Read image

```python
cv2.imread()
```

### Convert to grayscale

```python
cv2.cvtColor()
```

### Detect objects

```python
detectMultiScale()
```

### Draw rectangle

```python
cv2.rectangle()
```

### Display image

```python
cv2.imshow()
```

### Wait for keyboard input

```python
cv2.waitKey()
```

### Close windows

```python
cv2.destroyAllWindows()
```

---

# 🆚 Haar Cascade vs Deep Learning

| Feature            | Haar Cascade            | Deep Learning                     |
| ------------------ | ----------------------- | --------------------------------- |
| Approach           | Traditional ML          | Neural Networks                   |
| Training           | Pre-trained classifiers | Usually trained/fine-tuned models |
| Speed              | Fast                    | Can be computationally heavier    |
| Accuracy           | Moderate                | Generally higher                  |
| CPU usage          | Low                     | Higher                            |
| Easy for beginners | Yes                     | More complex                      |
| Robustness         | Limited                 | Usually better                    |
| Examples           | Face/Eye detection      | YOLO, CNN, RetinaFace             |

---

# ⚠️ Limitations

Haar Cascade is useful for learning and simple applications, but it has limitations:

* Sensitive to lighting
* Sensitive to face angle
* May produce false positives
* Less robust for occluded faces
* Detection quality depends on the cascade
* Modern deep-learning detectors generally handle difficult conditions better

---

# 🚀 Possible Future Improvements

This project can be extended into:

### 1. Webcam Face Detection

Use:

```python
cv2.VideoCapture(0)
```

to detect faces in real time.

### 2. Smile Detection

Add:

```text
haarcascade_smile.xml
```

### 3. Full-body Detection

Use:

```text
haarcascade_fullbody.xml
```

### 4. Face Detection with Streamlit

Create a simple web interface:

```text
Upload Image
      ↓
OpenCV
      ↓
Haar Cascade
      ↓
Face Detection
      ↓
Display Result
```

### 5. Face Detection with Gradio

Build a simple AI/CV interface using Gradio.

### 6. Modern Object Detection

Replace Haar Cascade with:

* YOLO
* SSD
* RetinaFace
* MTCNN
* MediaPipe

---

# 🎤 Interview Questions

## Beginner

### 1. What is Computer Vision?

Computer Vision is a field of AI that enables computers to understand and process images and videos.

### 2. What is OpenCV?

OpenCV is an open-source Computer Vision library used for image processing, video processing, object detection and many other computer vision tasks.

### 3. What is Haar Cascade?

Haar Cascade is a machine-learning-based object detection method that uses Haar-like features and a cascade of classifiers.

### 4. Why do we convert an image to grayscale?

To simplify the image and reduce computational complexity because detection can work with intensity information.

### 5. What is `detectMultiScale()`?

It detects objects at different scales within an image.

### 6. What is ROI?

ROI means **Region of Interest**. It is a selected portion of an image where we perform further processing.

---

## Intermediate

### 7. What is `scaleFactor`?

It determines the scaling step used when searching for objects at different sizes.

### 8. What is `minNeighbors`?

It determines how many neighboring detections are needed to validate an object detection.

### 9. Why detect eyes inside the face ROI?

Because eyes are expected to be located within the detected face. Searching only the face region reduces unnecessary computation and false detections.

### 10. What is a false positive?

A false positive occurs when the detector identifies an object where the object does not actually exist.

### 11. What is a false negative?

A false negative occurs when an actual object is present but the detector fails to detect it.

### 12. What are Haar-like features?

They are simple rectangular features that capture differences in brightness between neighboring regions.

---

## Advanced

### 13. What is AdaBoost?

AdaBoost is an ensemble learning technique that combines multiple weak learners to create a stronger classifier.

### 14. Why is an integral image used?

It allows rectangular pixel-region sums to be calculated efficiently.

### 15. What is a cascade classifier?

A cascade classifier contains multiple stages of classifiers. Simple stages reject obvious non-object regions quickly, while later stages perform more detailed checks.

### 16. Why can Haar Cascade fail?

It can struggle with:

* Rotation
* Poor lighting
* Occlusion
* Unusual face angles
* Low-quality images

### 17. Haar Cascade vs CNN?

Haar Cascade uses traditional feature-based detection, while CNN-based detectors learn visual features automatically from training data.

---

# 📄 requirements.txt

```text
opencv-python==4.10.0.84
numpy
```

Install:

```bash
python -m pip install -r requirements.txt
```

---

# 🔐 .gitignore

Do not upload your Python virtual environment.

Create `.gitignore`:

```text
.venv/
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
```

---

# ▶️ Run the Project

Activate the environment:

```bash
source .venv/Scripts/activate
```

Run:

```bash
python face_eye_detection.py
```

---

# 📌 Key Learning

This project demonstrates the basic Computer Vision pipeline:

```text
Image
  ↓
OpenCV
  ↓
Grayscale
  ↓
Haar Cascade
  ↓
Face Detection
  ↓
ROI
  ↓
Eye Detection
  ↓
Bounding Boxes
  ↓
Output
```

---

# 👨‍💻 Author

**Subrata Mondal**

Data Analyst | Data Science | Machine Learning | Artificial Intelligence | Computer Vision

---

# ⭐ Future Project Roadmap

```text
Haar Cascade
      ↓
Face & Eye Detection
      ↓
Webcam Detection
      ↓
Smile Detection
      ↓
Streamlit / Gradio
      ↓
Face Recognition
      ↓
CNN
      ↓
YOLO
      ↓
Advanced Computer Vision
```

---

## ⭐ If you find this project useful

Feel free to fork the repository, experiment with different Haar Cascade classifiers, and extend the project with real-time webcam detection.
