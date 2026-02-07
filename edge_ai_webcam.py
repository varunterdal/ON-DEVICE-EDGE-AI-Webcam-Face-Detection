import cv2
import numpy as np
import os
from collections import deque

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

FACE_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

AGE_PROTO = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_net.caffemodel")

GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")

# ===============================
# FILE CHECK
# ===============================
for f in [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]:
    if not os.path.exists(f):
        print("Missing file:", f)
        exit()

# ===============================
# LOAD MODELS
# ===============================
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60+)"
]
GENDER_BUCKETS = ["Male", "Female"]
MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# ===============================
# TEMPORAL SMOOTHING
# ===============================
age_history = deque(maxlen=7)
gender_history = deque(maxlen=7)

# ===============================
# OPEN CAMERA (WINDOWS SAFE)
# ===============================
cap = None
for i in range(3):
    cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cam.isOpened():
        cap = cam
        print("Camera opened at index", i)
        break

if cap is None:
    print("No webcam found")
    exit()

print("EDGE AI RUNNING — Press 'q' to quit")

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # -------- FACE DETECTION (DNN) --------
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Padding
            pad = 25
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                MEAN_VALUES, swapRB=False
            )

            # -------- GENDER --------
            gender_net.setInput(face_blob)
            gender_pred = gender_net.forward()
            gender = GENDER_BUCKETS[np.argmax(gender_pred)]
            gender_history.append(gender)

            # -------- AGE --------
            age_net.setInput(face_blob)
            age_pred = age_net.forward()
            age = AGE_BUCKETS[np.argmax(age_pred)]
            age_history.append(age)

            # -------- SMOOTHED RESULT --------
            final_gender = max(set(gender_history), key=gender_history.count)
            final_age = max(set(age_history), key=age_history.count)

            label = f"{final_gender}, {final_age}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, "EDGE AI : ON-DEVICE",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("EDGE AI - Face | Age | Gender", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
