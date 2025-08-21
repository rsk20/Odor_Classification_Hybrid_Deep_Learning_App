from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse


from backend.analyzer_module import SimpleModelPerformanceAnalyzer
from backend.train import train_model
from backend.evaluate import evaluate_model
from backend.data import load_data
from backend.logger import log_model_metrics, generate_model_id, get_model_save_path
from backend.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_loss_accuracy,
    plot_feature_importance
)
import tensorflow as tf
import os
import csv
from pydantic import BaseModel

app = FastAPI(
    title="Odor Classification API",
    description="Endpoints for training, prediction, and model management | Contact: rkilic20@epoka.edu.al",
    version="1.0.0"
)


class TrainRequest(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    model_name: str


def metrics_to_table(metrics):
    """Convert metrics dict to HTML table with enhanced styling"""
    return f"""
    <style>
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            margin: 10px 0;
        }}
        .metrics-table th {{
            background-color: #f2f2f2;
            padding: 8px;
            text-align: left;
            font-weight: bold;
        }}
        .metrics-table td {{
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }}
        .metrics-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metrics-value {{
            font-family: monospace;
            text-align: right;
        }}
    </style>
    <table class="metrics-table">
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td class="metrics-value">{metrics['accuracy']:.4f}</td>
        </tr>
        <tr>
            <td>Precision</td>
            <td class="metrics-value">{metrics['precision']:.4f}</td>
        </tr>
        <tr>
            <td>Recall</td>
            <td class="metrics-value">{metrics['recall']:.4f}</td>
        </tr>
        <tr>
            <td>F1-Score</td>
            <td class="metrics-value">{metrics['f1']:.4f}</td>
        </tr>
    </table>
    """


def generate_comparison_table():
    """Generate HTML table of all trained models from log file"""
    log_path = os.path.join('backend', 'model_log.csv')
    if not os.path.exists(log_path):
        return "<p>No model history available</p>"

    try:
        with open(log_path, mode='r') as file:
            models = list(csv.DictReader(file))

        if not models:
            return "<p>No models in history</p>"

        html = """
        <style>
            .comparison-table {
                width: 100%;
                border-collapse: collapse;
                font-family: Arial, sans-serif;
            }
            .comparison-table th {
                background-color: #f2f2f2;
                padding: 8px;
                text-align: left;
                font-weight: bold;
            }
            .comparison-table td {
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }
            .comparison-table tr:hover {
                background-color: #f5f5f5;
            }
            .numeric-value {
                font-family: monospace;
                text-align: right;
            }
            .model-name {
                max-width: 150px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
        </style>
        <table class="comparison-table">
            <tr>
                <th>Timestamp</th>
                <th>Model</th>
                <th>Accuracy</th>
                <th>F1</th>
                <th>Epochs</th>
                <th>Time (s)</th>
            </tr>
        """

        for model in sorted(models, key=lambda x: x['Timestamp'], reverse=True):
            html += f"""
            <tr>
                <td>{model['Timestamp'][:16]}</td>
                <td class="model-name" title="{model['Model Name']}">{model['Model Name'][:20]}{'...' if len(model['Model Name']) > 20 else ''}</td>
                <td class="numeric-value">{float(model['Accuracy']):.3f}</td>
                <td class="numeric-value">{float(model['F1-Score']):.3f}</td>
                <td class="numeric-value">{model['Epochs']}</td>
                <td class="numeric-value">{float(model['Training Time (s)']):.1f}</td>
            </tr>
            """

        html += "</table>"
        return html

    except Exception as e:
        return f"<p>Error loading comparison: {str(e)}</p>"


@app.post("/train_and_save_model")
async def train_and_save_model(req: TrainRequest):
    model_id = generate_model_id()
    save_path = get_model_save_path(model_id)

    # Train and get comprehensive results
    result = train_model(
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        save_path=save_path
    )
    # Save with custom name
    safe_name = "".join(c for c in req.model_name if c.isalnum() or c in ('_', '-')).rstrip()
    custom_model_path = os.path.join(os.path.dirname(result['save_path']), f"{safe_name}.h5")
    result['model'].save(custom_model_path)
    # Log metrics
    log_model_metrics(
        metrics=result['metrics'],
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        training_time_seconds=result['training_time'],
        model_id=model_id,
        model_name=safe_name,
        saved_model_filename=f"{safe_name}.h5"
    )
    # Generate visualizations
    y_pred, _ = evaluate_model(result['model'], result['X_test'], result['y_test'])
    class_labels = [str(i) for i in range(result['y_test'].shape[1])]

    visualizations = {
        'confusion_matrix': plot_confusion_matrix(result['y_test'], y_pred, class_labels),
        'roc_curve': plot_roc_curve(result['y_test'], y_pred, class_labels),
        'metrics_table': metrics_to_table(result['metrics']),  # Use our new HTML table
        'feature_importance': plot_feature_importance(),
        'model_comparison': generate_comparison_table()  # Add comparison table
    }
    visualizations.update(plot_loss_accuracy(result['history']))

    return JSONResponse({
        "message": f"Model '{safe_name}' trained and saved.",
        "metrics": result['metrics'],
        "visualizations": visualizations
    })


class ModelPathRequest(BaseModel):
    model_path: str

@app.post("/load_model_and_evaluate")
async def load_and_evaluate_model(request: ModelPathRequest):
    try:
        model_path = Path(request.model_path).resolve()
        if not model_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": "Model file not found"}
            )

        model = tf.keras.models.load_model(model_path)
        _, X_test, _, y_test = load_data()

        # Use centralized evaluation
        y_pred, metrics = evaluate_model(model, X_test, y_test)

        # Generate visualizations
        class_labels = [str(i) for i in range(y_test.shape[1])]
        visualizations = {
            'confusion_matrix': plot_confusion_matrix(y_test, y_pred, class_labels),
            'roc_curve': plot_roc_curve(y_test, y_pred, class_labels),
            'metrics_table': metrics_to_table(metrics),
            'model_comparison': generate_comparison_table()
        }

        return {
            "message": f"Model at '{model_path}' evaluated.",
            "metrics": metrics,
            "visualizations": visualizations
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/generate_analysis")
async def generate_model_analysis(request: ModelPathRequest):
    try:
        model_path = Path(request.model_path).resolve()
        if not model_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": "Model file not found"}
            )

        model = tf.keras.models.load_model(model_path)
        _, X_test, _, y_test = load_data()

        # Get metrics from evaluation
        y_pred, metrics = evaluate_model(model, X_test, y_test)

        # Add any additional metrics you want to include
        metrics['model_name'] = Path(model_path).stem

        # Generate the enhanced report
        analyzer = SimpleModelPerformanceAnalyzer(
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1=metrics['f1'],
            model_name=metrics.get('model_name', 'the model')
        )
        performance_report = analyzer.generate_report()

        return {
            "message": f"Model at '{model_path}' analyzed.",
            "metrics": metrics,
            "performance_report": performance_report
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})