# main.py
import cv2
from src.camera import open_camera
from src.client import EdgeClient
from src.federated import aggregate_updates
from config import WINDOW_NAME

# ---------------------------
# Simulate multiple clients
# ---------------------------
CLIENTS = [
    EdgeClient(client_id=1, low_resource=False),
    EdgeClient(client_id=2, low_resource=True),
    EdgeClient(client_id=3, low_resource=True)
]

# ---------------------------
# Open camera
# ---------------------------
cap = open_camera()
if not cap:
    exit()

print("EDGE AI CLIENTS SIMULATION — Press 'q' to quit")

frame_num = 0
round_id = 0
ROUND_INTERVAL = 30  # Every 30 frames

# ---------------------------
# Main loop
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # ---------------------------
    # Adaptive scheduling
    # ---------------------------
    for client in CLIENTS:
        client.check_resources(cpu_threshold=50)

    # ---------------------------
    # Federated aggregation
    # ---------------------------
    if frame_num % ROUND_INTERVAL == 0:
        round_id += 1
        print(f"[SERVER] Aggregation round {round_id}")
        aggregate_updates(CLIENTS, round_id)

    # ---------------------------
    # Display only (no client processing here)
    # ---------------------------
    cv2.putText(
        frame,
        f"EDGE AI FL SIMULATION | Round {round_id}",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------
# Cleanup
# ---------------------------
cap.release()
cv2.destroyAllWindows()
