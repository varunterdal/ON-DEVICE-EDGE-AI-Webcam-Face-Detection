import os
import csv
import time
import psutil
import numpy as np

# -------------------------------
# Ensure logs directory exists
# -------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


# -------------------------------
# CPU usage
# -------------------------------
def get_cpu_usage():
    """
    Returns current CPU usage percentage
    """
    return psutil.cpu_percent(interval=0.1)


# -------------------------------
# Measure latency of a function
# -------------------------------
def measure_latency(func, *args, **kwargs):
    """
    Measures execution time of a function
    """
    start = time.time()
    result = func(*args, **kwargs)
    latency = time.time() - start
    return result, latency


# -------------------------------
# CSV logging utility
# -------------------------------
def log_to_csv(
    file_path,
    round_id,
    client_id,
    dataset_size,
    staleness,
    local_accuracy,
    update_norm,
    skipped,
    cpu,
    latency
):
    """
    Logs FL-related metrics to CSV
    """

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header only once
        if not file_exists:
            writer.writerow([
                "round",
                "client_id",
                "dataset_size",
                "staleness",
                "local_accuracy",
                "update_norm",
                "skipped",
                "cpu_usage",
                "latency"
            ])

        writer.writerow([
            round_id,
            client_id,
            dataset_size,
            staleness,
            round(local_accuracy, 4),
            round(update_norm, 4),
            skipped,
            round(cpu, 2),
            round(latency, 4)
        ])
# At the start of the run, overwrite previous logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

for client_id in [1, 2, 3]:
    log_file = os.path.join(LOG_DIR, f"client_{client_id}_metrics.csv")
    if os.path.exists(log_file):
        os.remove(log_file)
