import cv2

face_path = r"D:\work\olama resm\Computer Vision\Haar cascade classifier\Haarcascades\haarcascade_frontalface_default.xml"
eye_path = r"D:\work\olama resm\Computer Vision\Haar cascade classifier\Haarcascades\haarcascade_eye.xml"
image_path = r"D:\work\olama resm\Computer Vision\Haar cascade classifier\faceinput.jpg"

# Load classifiers
face_classifier = cv2.CascadeClassifier(face_path)
eye_classifier = cv2.CascadeClassifier(eye_path)

# Check classifiers
if face_classifier.empty():
    print("❌ Face Haar Cascade not loaded")
    exit()

if eye_classifier.empty():
    print("❌ Eye Haar Cascade not loaded")
    exit()

print("✅ Haar Cascade files loaded")


# Load image
img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found")
    exit()

print("✅ Image loaded")


# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Detect faces
faces = face_classifier.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5
)

print("Faces detected:", len(faces))


# Process each face
for (x, y, w, h) in faces:

    # Draw face rectangle
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (127, 0, 255),
        2
    )

    # Face ROI
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # Detect eyes
    eyes = eye_classifier.detectMultiScale(roi_gray)

    print("Eyes detected:", len(eyes))

    for (ex, ey, ew, eh) in eyes:

        cv2.rectangle(
            roi_color,
            (ex, ey),
            (ex + ew, ey + eh),
            (255, 255, 0),
            2
        )


# Display result
cv2.imshow("Face and Eye Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
