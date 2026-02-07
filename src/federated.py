import csv
import os
import time
import numpy as np

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

SERVER_LOG = os.path.join(LOG_DIR, "server_global_metrics.csv")

def aggregate_updates(clients, round_id):
    updates = []
    weights = []

    for c in clients:
        if not c.has_update:
            continue

        # Async FL weight
        w = (c.dataset_size * c.local_accuracy) / (1 + c.staleness)
        updates.append(c.update_norm)
        weights.append(w)

        c.has_update = False  # mark consumed

    if not updates:
        return

    updates = np.array(updates)
    weights = np.array(weights)

    # -------- Global metrics --------
    global_mse = np.average(updates ** 2, weights=weights)
    global_accuracy = np.average(
        [c.local_accuracy for c in clients if c.local_accuracy > 0]
    )

    log_server_metrics(round_id, global_accuracy, global_mse)

def log_server_metrics(round_id, acc, mse):
    file_exists = os.path.exists(SERVER_LOG)

    with open(SERVER_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "round",
                "global_accuracy",
                "global_mse",
                "server_time"
            ])
        writer.writerow([
            round_id,
            round(acc, 4),
            round(mse, 6),
            time.time()
        ])
