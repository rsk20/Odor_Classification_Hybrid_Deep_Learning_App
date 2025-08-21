import time
import os
from backend.logger import log_model_metrics, generate_model_id, get_model_save_path
from backend.train import train_model
from backend.evaluate import evaluate_model
from backend.data import load_data
from backend.visualization import plot_loss_accuracy


def get_user_input():
    try:
        epochs = int(input("Enter number of epochs: "))
        batch_size = int(input("Enter batch size: "))
        learning_rate = float(input("Enter learning rate (e.g., 0.001): "))
        return epochs, batch_size, learning_rate
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
        return None, None, None


def get_model_name():
    model_name = input("Enter a name for your model: ").strip()
    if not model_name:
        model_name = "untitled_model"  # Default name if no input provided
    return model_name


if __name__ == "__main__":
    print("=== Odor Classification Model Training ===")

    # Get all user input first
    epochs, batch_size, learning_rate = get_user_input()
    if None in (epochs, batch_size, learning_rate):
        print("Aborting due to invalid input.")
        exit(1)

    model_name = get_model_name()
    model_id = generate_model_id()

    # Train model
    start_time = time.time()
    model, history = train_model(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_path=None
    )
    training_time = time.time() - start_time

    # Plot training loss and accuracy
    plot_loss_accuracy(history)

    # Defensive retrieval of accuracy values
    train_accuracy = history.history.get('accuracy', [None])[-1]
    val_accuracy = history.history.get('val_accuracy', [None])[-1]
    if train_accuracy is not None and val_accuracy is not None:
        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
    else:
        print("\nTraining or validation accuracy not available.")

    # Load test data for evaluation
    _, X_test, _, y_test = load_data()

    # Evaluate model and show plots
    y_pred, metrics = evaluate_model(model, X_test, y_test, plot=True)

    # Save model with custom name
    model_name_path = os.path.join(os.path.dirname(get_model_save_path(model_id)), f"{model_name}.h5")
    model.save(model_name_path)
    print(f"\nModel saved as: {model_name_path}")

    log_model_metrics(
        metrics=metrics,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_time_seconds=training_time,
        model_id=model_id,
        model_name=model_name,
        saved_model_filename=f"{model_name}.h5"
    )

    print(f"\nTraining complete. Model saved at: {model_name_path}")