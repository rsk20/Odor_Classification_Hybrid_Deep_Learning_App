import csv
import os
from datetime import datetime
import uuid
from collections import OrderedDict
from backend.config import MODEL_LOG_PATH, MODEL_DIR

def generate_model_id():
    return str(uuid.uuid4())[:8]  # Short unique ID

def get_model_save_path(model_id):
    os.makedirs(MODEL_DIR, exist_ok=True)
    return os.path.join(MODEL_DIR, f"model_{model_id}.h5")

def log_model_metrics(metrics, epochs, batch_size, learning_rate, training_time_seconds, model_id,
                     model_name="", saved_model_filename="", csv_path=MODEL_LOG_PATH):

    required_metrics = ["accuracy", "precision", "recall", "f1"]
    for metric in required_metrics:
        if metric not in metrics:
            raise ValueError(f"Missing required metric: {metric}")
    try:
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else None, exist_ok=True)
        file_exists = os.path.isfile(csv_path)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ensure empty strings instead of None
        model_name = model_name or ""
        saved_model_filename = saved_model_filename or ""

        field_order = [
        "Model_ID",
        "Timestamp",
        "Model Name",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "Epochs",
        "Batch Size",
        "Learning Rate",
        "Training Time (s)",
        "Saved Model"
        ]

        row = {
            "Model_ID": model_id,
            "Timestamp": timestamp,
            "Model Name": model_name,
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1-Score": round(metrics["f1"], 4),
            "Epochs": epochs,
            "Batch Size": batch_size,
            "Learning Rate": learning_rate,
            "Training Time (s)": round(training_time_seconds, 2),
            "Saved Model": saved_model_filename
        }

        # Debug output
        print("\nDEBUG - Writing to CSV:")
        print(f"Field order: {field_order}")
        print(f"Row data: {row}")

        with open(csv_path, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_order)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    except Exception as e:
        print(f"\nERROR in logging: {str(e)}")
        raise