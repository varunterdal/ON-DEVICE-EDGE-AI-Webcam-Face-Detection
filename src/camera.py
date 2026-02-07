# src/camera.py
import cv2
from config import CAMERA_MAX_INDEX

def open_camera():
    cap = None
    for i in range(CAMERA_MAX_INDEX):
        cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cam.isOpened():
            cap = cam
            print("Camera opened at index", i)
            break
    if cap is None:
        print("No webcam found")
    return cap
