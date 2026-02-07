import time
import random
import numpy as np

from src.face_detector import FaceDetector
from src.age_gender import AgeGenderPredictor
from src.utils import (
    get_cpu_usage,
    measure_latency,
    log_to_csv
)


class EdgeClient:
    def __init__(self, client_id, low_resource=False):
        self.client_id = client_id
        self.low_resource = low_resource

        # Models
        self.face_detector = FaceDetector()
        self.predictor = AgeGenderPredictor(partial=low_resource)

        # FL-related attributes
        self.dataset_size = random.randint(80, 300)   # simulated local data
        self.local_accuracy = 0.0
        self.update_norm = 0.0
        self.staleness = 0
        self.has_update = False
        self.active = True

        # Logging
        self.log_file = f"logs/client_{self.client_id}_metrics.csv"

    # -------------------------------
    # Resource check (Adaptive scheduling)
    # -------------------------------
    def check_resources(self, cpu_threshold=50):
        cpu = get_cpu_usage()

        if cpu > cpu_threshold and self.low_resource:
            self.active = False
            self.staleness += 1
        else:
            self.active = True

        return self.active

    # -------------------------------
    # Process frame (edge inference)
    # -------------------------------
    def process_frame(self, frame, round_id):
        if not self.active:
            return frame

        start_time = time.time()

        faces = self.face_detector.detect_faces(frame)

        correct = 0
        total = 0

        for (x1, y1, x2, y2, _) in faces:
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            (gender, age), latency = measure_latency(
                self.predictor.predict, face
            )

            # ---- Simulated correctness ----
            # (we don't have true labels)
            if random.random() > 0.3:
                correct += 1
            total += 1

        # -------------------------------
        # Simulated local accuracy
        # -------------------------------
        if total > 0:
            self.local_accuracy = correct / total
        else:
            self.local_accuracy = random.uniform(0.6, 0.9)

        # -------------------------------
        # Simulated model update
        # -------------------------------
        self.update_norm = abs(
            np.random.normal(
                loc=0.5 if self.low_resource else 1.0,
                scale=0.1
            )
        )

        self.has_update = True
        self.staleness = 0  # reset after update

        cpu = get_cpu_usage()
        latency_total = time.time() - start_time

        # -------------------------------
        # Log client metrics
        # -------------------------------
        log_to_csv(
            self.log_file,
            round_id=round_id,
            client_id=self.client_id,
            dataset_size=self.dataset_size,
            staleness=self.staleness,
            local_accuracy=self.local_accuracy,
            update_norm=self.update_norm,
            skipped=0,
            cpu=cpu,
            latency=latency_total
        )

        return frame
