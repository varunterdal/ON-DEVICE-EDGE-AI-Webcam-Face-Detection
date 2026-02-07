# 🧠 ON-DEVICE EDGE AI: Webcam Face Detection

## 📌 Project Overview
This project demonstrates a **real-time, on-device Edge AI application** that performs **face detection using a laptop webcam**.  
The entire system runs **locally on the device (edge)** without using cloud services, highlighting the core benefits of **Edge AI** such as low latency, privacy, and offline operation.

A **Haar Cascade Classifier** from OpenCV is used for lightweight and efficient face detection, making it suitable for edge devices like laptops and embedded systems.

---

## 🎯 Key Features
- 📷 Real-time face detection using laptop webcam  
- 🧠 Fully on-device Edge AI processing  
- ⚡ Lightweight classical computer vision model  
- 🟩 Bounding boxes around detected faces  
- 💾 Option to save detected frames  
- 🔗 Optional FastAPI backend integration  
- 🖥️ Cross-platform support (Windows, macOS, Linux)

---

## 🛠️ Technologies Used
- **Python 3.8+**
- **OpenCV (cv2)**
- **Haar Cascade Classifier**
- **FastAPI** (optional)
- **curl** (optional)

---

edge_ai_project/
│
├── edge_ai_webcam.py # Main face detection script
├── captured.jpg # Saved frame (generated at runtime)
├── fastapi_receiver.py # Optional FastAPI backend
├── README.md # Project documentation
└── venv/ # Virtual environment (optional)

---

## ⚙️ Installation & Setup

### 1️⃣ Create Project Folder
```bash
cd Desktop
mkdir edge_ai_project
cd edge_ai_project

---

## ⚙️ Installation & Setup

### 1️⃣ Create Project Folder
```bash
cd Desktop
mkdir edge_ai_project
cd edge_ai_project
2️⃣ (Optional) Create & Activate Virtual Environment

Windows

python -m venv venv
.\venv\Scripts\activate


macOS / Linux

python3 -m venv venv
source venv/bin/activate

3️⃣ Install Required Libraries
pip install numpy==1.26.4 --only-binary=:all:
pip install opencv-python --only-binary=:all:

🧾 Python Script (edge_ai_webcam.py)
import cv2

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open laptop webcam (edge device)
cap = cv2.VideoCapture(0)

print("Running ON-DEVICE EDGE AI... Press 'q' to quit, 's' to save frame")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("ON-DEVICE EDGE AI (Laptop Webcam)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite("captured.jpg", frame)
        print("Frame saved as captured.jpg")

cap.release()
cv2.destroyAllWindows()

▶️ Running the Application
python edge_ai_webcam.py

🟢 Expected Output

Webcam turns ON

Green rectangles appear around detected faces

Press s to save an image

Press q to quit the application

🌐 Optional: FastAPI Backend Integration
Install FastAPI
pip install fastapi uvicorn python-multipart

Run Backend Server
uvicorn fastapi_receiver:app --reload

Upload Image Using curl
curl -X POST "http://127.0.0.1:8000/compare" `
     -H "Content-Type: multipart/form-data" `
     -F "file=@captured.jpg"

Swagger UI
http://127.0.0.1:8000/docs

🧪 Troubleshooting
❌ NumPy / OpenCV installation error
pip install numpy==1.26.4 --only-binary=:all:
pip install opencv-python --only-binary=:all:

❌ Webcam not opening

Close Zoom / Teams / Browser

Check OS camera permissions

Try changing camera index:

cv2.VideoCapture(1)

❌ Faces not detected properly

Improve lighting

Face the camera directly

Haar cascades are sensitive to angle and illumination

🚀 Future Enhancements

DNN-based face detector (higher accuracy)

Face recognition using embeddings

ESP32-CAM integration

Packaging as executable using PyInstaller

Cloud or IoT backend deployment

📚 Learning Outcomes

Understanding Edge AI concepts

Real-time computer vision implementation

On-device AI deployment

Webcam interfacing with OpenCV

Backend integration using FastAPI

📜 License

This project is intended for educational and academic use.
Free to modify and extend.


---
