# src/face_detector.py
import cv2
import numpy as np
from config import FACE_PROTO, FACE_MODEL, CONF_THRESHOLD

class FaceDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONF_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2, y2, confidence))
        return faces
