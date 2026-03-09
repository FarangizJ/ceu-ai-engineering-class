import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------------------------------
# Load acne detection model
# ------------------------------------------------

acne_model = YOLO("chatbot/models/best.pt")

# ------------------------------------------------
# OpenCV face detector
# ------------------------------------------------

face_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ------------------------------------------------
# Detect face
# ------------------------------------------------

def detect_face(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    return img[y:y+h, x:x+w]

# ------------------------------------------------
# Acne detection
# ------------------------------------------------

def detect_acne(face):

    results = acne_model(face)

    acne_count = 0

    for r in results:
        acne_count += len(r.boxes)

    return acne_count

# ------------------------------------------------
# Oil detection
# ------------------------------------------------

def oil_detection(face):

    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

    brightness = hsv[:, :, 2]

    mean_brightness = np.mean(brightness)

    if mean_brightness > 160:
        return "high"
    elif mean_brightness > 120:
        return "moderate"
    else:
        return "low"

# ------------------------------------------------
# Redness detection
# ------------------------------------------------

def redness_detection(face):

    b, g, r = cv2.split(face)

    redness_index = np.mean(r / (g + b + 1))

    if redness_index > 0.9:
        return "high"
    elif redness_index > 0.7:
        return "moderate"
    else:
        return "low"

# ------------------------------------------------
# Pore estimation
# ------------------------------------------------

def pore_estimation(face):

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    variance = laplacian.var()

    if variance > 200:
        return "large"
    elif variance > 100:
        return "moderate"
    else:
        return "small"

# ------------------------------------------------
# Skin type classification
# ------------------------------------------------

def determine_skin_type(oil_level):

    if oil_level == "high":
        return "oily"

    if oil_level == "moderate":
        return "combination"

    return "normal"

# ------------------------------------------------
# Main analysis
# ------------------------------------------------

def analyze_skin(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return "Image could not be loaded"

    face = detect_face(img)

    if face is None:
        return "Face not detected"

    acne = detect_acne(face)
    oil = oil_detection(face)
    redness = redness_detection(face)
    pores = pore_estimation(face)

    skin_type = determine_skin_type(oil)

    return f"""
Skin Analysis

Skin Type: {skin_type}

Metrics
Acne spots detected: {acne}
Oil level: {oil}
Redness: {redness}
Pore visibility: {pores}
"""