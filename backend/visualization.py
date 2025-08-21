from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import base64
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_to_base64(fig):
    """Convert matplotlib figure to base64 encoded image"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def plot_confusion_matrix(y_true, y_pred, class_labels):
    y_true_labels = y_true.argmax(axis=1)
    y_pred_labels = y_pred.argmax(axis=1)
    cm = confusion_matrix(y_true_labels, y_pred_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    return plot_to_base64(fig)


def plot_roc_curve(y_true, y_pred, class_labels):
    y_true_bin = y_true
    y_score = y_pred
    n_classes = y_true.shape[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'Class {class_labels[i]} (area = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    return plot_to_base64(fig)


def plot_loss_accuracy(history):
    # Loss plot
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss vs Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    loss_img = plot_to_base64(fig1)

    # Accuracy plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Accuracy vs Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    acc_img = plot_to_base64(fig2)

    return {'loss_vs_epochs': loss_img, 'accuracy_vs_epochs': acc_img}


def plot_metrics_table(metrics):
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis('off')

    data = [
        ['Accuracy', f"{metrics['accuracy']:.4f}"],
        ['Precision', f"{metrics['precision']:.4f}"],
        ['Recall', f"{metrics['recall']:.4f}"],
        ['F1 Score', f"{metrics['f1']:.4f}"]
    ]

    table = ax.table(cellText=data, colLabels=['Metric', 'Value'], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    return plot_to_base64(fig)


def plot_feature_importance(feature_importances=None, feature_names=None):
    if feature_importances is None:
        feature_importances = np.random.rand(10)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(feature_importances))]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=feature_importances, y=feature_names, ax=ax)
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    return plot_to_base64(fig)