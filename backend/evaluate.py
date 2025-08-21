# backend/evaluate.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from backend.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_metrics_table,
    plot_feature_importance
)


def evaluate_model(model, X_test, y_test, plot=False):
    """Centralized evaluation with validation and consistent metrics"""
    # Convert predictions to class indices
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = y_pred_prob.argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    # Validation
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    # Consistent metric calculation
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    if plot:
        class_labels = [str(i) for i in range(y_test.shape[1])]
        plot_confusion_matrix(y_test, y_pred_prob, class_labels)
        plot_roc_curve(y_test, y_pred_prob, class_labels)
        plot_metrics_table(metrics)
        plot_feature_importance()

    print(classification_report(y_true, y_pred))

    return y_pred_prob, metrics  # Return probabilities for visualization