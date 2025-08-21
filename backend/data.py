import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from backend.config import PROCESSED_DATA_PATH, PROCESSED_LABELS_PATH


def load_data():
    data = np.load(PROCESSED_DATA_PATH)
    labels = np.load(PROCESSED_LABELS_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels.argmax(axis=1)
    )

    scaler = StandardScaler()
    num_samples, num_sensors, num_timesteps = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_sensors * num_timesteps)
    X_test_reshaped = X_test.reshape(-1, num_sensors * num_timesteps)

    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

    X_train_scaled = X_train_scaled.transpose(0, 2, 1)
    X_test_scaled = X_test_scaled.transpose(0, 2, 1)

    return X_train_scaled, X_test_scaled, y_train, y_test

def load_test_data(path=None):
    if path is not None:
        data = np.load(path)
        X_test = data['X']
        y_test = data['y']
    else:
        _, X_test, _, y_test = load_data()

    scaler = StandardScaler()
    num_samples, num_timesteps, num_sensors = X_test.shape
    X_test_reshaped = X_test.reshape(-1, num_timesteps * num_sensors)
    X_test_scaled = scaler.fit_transform(X_test_reshaped).reshape(X_test.shape)
    X_test_scaled = X_test_scaled.transpose(0, 2, 1)

    return X_test_scaled, y_test