# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Face Detection
FACE_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Age
AGE_PROTO = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_net.caffemodel")
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60+)"]

# Gender
GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")
GENDER_BUCKETS = ["Male", "Female"]

# Image preprocessing
MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Detection parameters
CONF_THRESHOLD = 0.6
PAD = 25

# Temporal smoothing
SMOOTHING_WINDOW = 7

# Video settings
CAMERA_MAX_INDEX = 3
WINDOW_NAME = "EDGE AI - Face | Age | Gender"
