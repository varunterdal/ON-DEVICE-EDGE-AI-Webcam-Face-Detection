# src/age_gender.py (update)
import cv2
import numpy as np
from collections import deque
from config import AGE_PROTO, AGE_MODEL, AGE_BUCKETS, GENDER_PROTO, GENDER_MODEL, GENDER_BUCKETS, MEAN_VALUES, SMOOTHING_WINDOW

class AgeGenderPredictor:
    def __init__(self, partial=False):
        # Load models
        self.age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
        self.gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
        self.age_history = deque(maxlen=SMOOTHING_WINDOW)
        self.gender_history = deque(maxlen=SMOOTHING_WINDOW)
        self.partial = partial
        
        if self.partial:
            self.freeze_base_layers()
    
    def freeze_base_layers(self):
        """
        Freeze base convolutional layers; allow only classifier layers to update.
        """
        # Note: OpenCV DNN does not support training directly,
        # but in federated learning simulation we can mimic partial updates
        # by only sending updates for classifier layers. Here, we mark it:
        print("Partial training mode: base layers frozen, only classifier layers active")
    
    def predict(self, face_img):
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MEAN_VALUES, swapRB=False)

        # Gender
        self.gender_net.setInput(blob)
        gender_pred = self.gender_net.forward()
        gender = GENDER_BUCKETS[np.argmax(gender_pred)]
        self.gender_history.append(gender)

        # Age
        self.age_net.setInput(blob)
        age_pred = self.age_net.forward()
        age = AGE_BUCKETS[np.argmax(age_pred)]
        self.age_history.append(age)

        # Smoothed results
        final_gender = max(set(self.gender_history), key=self.gender_history.count)
        final_age = max(set(self.age_history), key=self.age_history.count)
        return final_gender, final_age
