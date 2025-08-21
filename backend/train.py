import time
import numpy as np
from backend.data import load_data
from backend.model import build_model
from backend.logger import generate_model_id, get_model_save_path
from backend.evaluate import evaluate_model  # Use centralized evaluation

def train_model(epochs, batch_size, learning_rate, save_path=None):
    """Train model and return comprehensive results"""
    X_train, X_test, y_train, y_test = load_data()
    num_timesteps, num_sensors = X_train.shape[1], X_train.shape[2]
    num_classes = y_train.shape[1]

    model = build_model(num_timesteps, num_sensors, num_classes, learning_rate)

    # Training
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    training_time = time.time() - start_time

    # Centralized evaluation
    y_pred, metrics = evaluate_model(model, X_test, y_test)

    # Generate model ID and save
    model_id = generate_model_id()
    if not save_path:
        save_path = get_model_save_path(model_id)
    model.save(save_path)

    return {
        'model': model,
        'history': history,
        'metrics': metrics,
        'model_id': model_id,
        'training_time': training_time,
        'save_path': save_path,
        'X_test': X_test,  # For visualization
        'y_test': y_test   # For visualization
    }