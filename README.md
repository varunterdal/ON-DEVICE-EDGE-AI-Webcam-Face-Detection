# 🧠 EDGE AI PROJECT – Real-Time Face, Age & Gender Detection

## 📌 Project Description

This project implements a **complete On-Device Edge AI system** that performs **real-time face detection, age prediction, and gender classification** using a laptop webcam.
All inference is executed **locally on the edge device**, without relying on cloud services, ensuring **low latency, privacy, and offline functionality**.

The system uses **OpenCV DNN models (Caffe-based)** for accurate face detection and age/gender estimation, making it suitable for **Edge AI, Computer Vision, and IoT demonstrations**.

---

## 🎯 Key Features

* 📷 Real-time webcam-based face detection
* 🧑 Age estimation of detected faces
* 🚻 Gender classification (Male / Female)
* 🧠 Fully on-device Edge AI (no cloud dependency)
* ⚡ DNN-based models for better accuracy
* 🧩 Modular project structure
* 🔌 Extensible for federated learning & client-server use

---

## 🛠️ Technologies Used

* **Python 3.8+**
* **OpenCV (cv2 + DNN module)**
* **Caffe Pre-trained Models**
* **NumPy**
* **FastAPI / Client scripts (optional)**
* **Federated learning concepts (experimental)**

---

## 📂 Project Structure

```
EDGE_AI_PROJECT/
│
├── __pycache__/
├── logs/
├── models/
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── deploy.prototxt
│   ├── gender_deploy.prototxt
│   ├── gender_net.caffemodel
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── age_gender.py          # Age & gender prediction logic
├── camera.py              # Webcam handling
├── client.py              # Client-side communication
├── config.py              # Configuration parameters
├── edge_ai_webcam.py      # Basic edge AI webcam demo
├── face_detector.py       # DNN-based face detection
├── federated.py           # Federated learning logic (experimental)
├── utils.py               # Helper functions
├── main.py                # Main application entry point
│
├── venv/                  # Python virtual environment
│
└── .gitignore
```

---

## ⚙️ Installation & Setup

### Step 1: Create & Activate Virtual Environment

**Windows**

```
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux**

```
python3 -m venv venv
source venv/bin/activate
```

---

### Step 2: Install Required Libraries

```
pip install numpy==1.26.4 --only-binary=:all:
pip install opencv-python --only-binary=:all:
```

(Optional, if using networking or APIs)

```
pip install fastapi uvicorn requests
```

---

## ▶️ How to Run the Project

### Run Full Edge AI Pipeline

```
python main.py
```

### Run Basic Webcam Face Detection Only

```
python edge_ai_webcam.py
```

---

## 🖥️ Expected Output

* Webcam activates
* Faces detected with bounding boxes
* Age and gender displayed near detected faces
* Real-time inference on edge device

---

## 🧠 Models Used

* **Face Detection:**
  `res10_300x300_ssd_iter_140000.caffemodel`
* **Age Prediction:**
  `age_net.caffemodel`
* **Gender Classification:**
  `gender_net.caffemodel`

All models are loaded from the `models/` directory using OpenCV’s DNN module.

---

## 🧪 Troubleshooting

### NumPy / OpenCV Installation Error

```
pip install numpy==1.26.4 --only-binary=:all:
pip install opencv-python --only-binary=:all:
```

### Webcam Not Opening

* Close Zoom / Teams / Browser
* Check OS camera permissions
* Try changing camera index in `camera.py`:

```python
cv2.VideoCapture(1)
```

### Poor Detection Accuracy

* Improve lighting conditions
* Face the camera directly
* Maintain reasonable distance from webcam

---

## 🚀 Future Enhancements

* Face recognition using embeddings
* ESP32-CAM integration
* Model optimization for embedded devices
* Cloud + Edge hybrid deployment
* Full federated learning implementation
* Packaging as executable using PyInstaller

---

## 📚 Learning Outcomes

* Understanding Edge AI concepts
* Real-time computer vision using DNNs
* On-device AI deployment
* Modular AI system design
* Practical use of OpenCV DNN models

---

## 📜 License

This project is intended for **educational and academic use only**.
Free to modify and extend for learning and research purposes.

