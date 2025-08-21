import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


DATASET_DIR = os.path.join(DATA_DIR, "gas_sensor_array_dataset/")

PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.npy")
PROCESSED_LABELS_PATH = os.path.join(DATA_DIR, "processed_labels.npy")


# Log path
MODEL_LOG_PATH = os.path.join(BASE_DIR, "model_log.csv")

PATH_RESULT = BASE_DIR / "Results"
#PATH_RESULT = os.path.join(BASE_DIR, "Results")
