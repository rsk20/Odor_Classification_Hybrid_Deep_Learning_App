import os

from backend.config import MODEL_LOG_PATH

os.environ['QT_DEBUG_PLUGINS'] = '1'

try:
    import ctypes

    try:
        ctypes.CDLL('msvcp140.dll')
    except OSError:
        print("Warning: Could not load msvcp140.dll - ensure VC++ Redistributable is installed")
except ImportError:
    pass

import logging

logging.basicConfig(filename='app_errors.log', level=logging.ERROR)

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QTextEdit, QLineEdit, QSpinBox,
    QDoubleSpinBox, QMessageBox, QFileDialog, QScrollArea, QSizePolicy, QDialog
)
from base64 import b64decode
import requests
import csv
import sys
from pathlib import Path
from PyQt5.QtGui import QPixmap, QIcon
import logging


class TrainModelThread(QThread):
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, api_url, params):
        super().__init__()
        self.api_url = api_url
        self.params = params
        self.setTerminationEnabled(True)

    def run(self):
        try:
            self.log_signal.emit(f"Starting training: {self.params}")
            response = requests.post(
                self.api_url,
                json=self.params,
                headers={'Content-Type': 'application/json'},
                timeout=1000
            )

            if self.isInterruptionRequested():
                return

            if response.status_code == 200:
                self.finished_signal.emit(response.json())
                self.log_signal.emit("\n✅ Training completed successfully!")
            else:
                error = f"❌ Error {response.status_code}: {response.text}"
                self.error_signal.emit(error)
                self.log_signal.emit(error)

        except Exception as e:
            msg = f"❌ Exception: {str(e)}"
            self.error_signal.emit(msg)
            self.log_signal.emit(msg)
        finally:
            self.deleteLater()


class EvaluateModelThread(QThread):
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, api_url, model_path):
        super().__init__()
        self.base_url = api_url.rstrip('/')
        self.model_path = model_path
        self.setTerminationEnabled(True)

    def run(self):
        try:
            combined_result = {}

            payload = {"model_path": self.model_path}


            self.log_signal.emit("\nLoading and evaluating model...")
            eval_response = requests.post(
                f"{self.base_url}/load_model_and_evaluate",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=1000
            )
            eval_response.raise_for_status()
            combined_result.update(eval_response.json())


            self.log_signal.emit("\nGenerating performance analysis...")
            analysis_response = requests.post(
                f"{self.base_url}/generate_analysis",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=1000
            )
            analysis_response.raise_for_status()
            combined_result.update(analysis_response.json())

            self.finished_signal.emit(combined_result)

        except requests.exceptions.RequestException as e:
            if e.response is not None:
                msg = f"❌ HTTP Error {e.response.status_code}: {e.response.text}"
            else:
                msg = f"❌ Network Error: {str(e)}"
            self.error_signal.emit(msg)
            self.log_signal.emit(msg)
        except Exception as e:
            msg = f"❌ Evaluation failed: {str(e)}"
            self.error_signal.emit(msg)
            self.log_signal.emit(msg)


class OdorClassificationUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Odor Classification System")
        self.setGeometry(50, 50, 1600, 900)
        self.setWindowIcon(QIcon("logo_odor_app.png"))


        self.api_url = "http://127.0.0.1:8000"
        self.current_model_path = None


        self.init_stylesheets()
        self.init_ui()
        self.load_model_comparison()

    def init_stylesheets(self):
        self.light_stylesheet = """
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11pt;
                color: #222222;
                background: #F9F9F9;
            }
            QLabel {
                font-weight: bold;
                color: #041453;
                background-color: transparent;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background: white;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 6px 10px;
                color: #222222;
                selection-background-color: #357ABD;
                selection-color: white;
            }
            QTextEdit {
                background: white;
                border: 1px solid #ccc;
                border-radius: 5px;
                color: #222222;
            }
            QGroupBox {
                border: 1px solid #ccc;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: 500;
                padding: 8px 12px 12px 12px;
                background: white;
            }
            QGroupBox QLabel {
                color: #041453;
;
            }
            QScrollArea {
                border: 1px solid #ccc;
                border-radius: 4px;
                background: white;
            }
        """

        self.viz_light_style = """
            QLabel {
                font-style: italic;
                color: #999;
                padding: 20px;
                background: white;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """

        self.title_light_style = """
            QLabel {
                font-weight: bold;
                padding: 4px 0;
                color: #333;
            }
        """

        self.setStyleSheet(self.light_stylesheet)

    def init_ui(self):
        outer_layout = QVBoxLayout()
        outer_layout.setSpacing(12)
        outer_layout.setContentsMargins(16, 16, 16, 16)

        # Create header
        self.create_header(outer_layout)

        # Main content layout
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)

        # Create panels
        main_layout.addLayout(self.create_sidebar(), 2)
        main_layout.addLayout(self.create_center_panel(), 6)
        main_layout.addLayout(self.create_right_panel(), 2)

        outer_layout.addLayout(main_layout)
        self.setLayout(outer_layout)

    def create_header(self, parent_layout):
        header_widget = QWidget()
        header_widget.setStyleSheet("background-color: #041453; font-size: 30px; font-weight: bold; border-radius: 8px; color: white;")

        header_inner_layout = QHBoxLayout(header_widget)
        header_inner_layout.setContentsMargins(12, 6, 12, 6)  # Keep some side margins
        header_inner_layout.setSpacing(0)  # Set spacing between widgets to 0

        # Load logo with error handling
        logo = QPixmap("new_logo2.png")
        if not logo.isNull():
            logo = logo.scaled(80, 80, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            logo_label = QLabel()
            logo_label.setPixmap(logo)
            logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            logo_label.setStyleSheet("margin: 0; padding: 0;")  # Remove all spacing
            header_inner_layout.addWidget(logo_label)
        else:
            placeholder = QLabel("LOGO")
            placeholder.setStyleSheet("""
                font-size: 24px; 
                font-weight: bold;
                margin: 0;
                padding: 0;
                color: white;
            """)
            header_inner_layout.addWidget(placeholder)

        title_label = QLabel("  ODOR CLASSIFICATION SYSTEM")
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18pt;
                font-weight: bold;
                margin: 0;
                padding: 0;
                padding-left: 10px;  # Small padding if needed
            }
        """)
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header_inner_layout.addWidget(title_label)

        # Add stretch to push everything to the left
        header_inner_layout.addStretch()

        parent_layout.addWidget(header_widget)

    def modern_button_style(self, button, bg_color="#0f2740", fg_color="white"):
        def darker(color, amount=30):
            c = color.lstrip('#')
            r = max(0, int(c[0:2], 16) - amount)
            g = max(0, int(c[2:4], 16) - amount)
            b = max(0, int(c[4:6], 16) - amount)
            return f"#{r:02x}{g:02x}{b:02x}"

        hover_color = darker(bg_color, 15)
        pressed_color = darker(bg_color, 40)

        button.setMinimumHeight(36)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: {fg_color};
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 11pt;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
            }}
        """)

    def create_sidebar(self):
        """Create the sidebar with model control buttons and analysis"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(8, 8, 8, 8)

        # Model status label
        self.loaded_model_label = QLabel("Loaded: No model selected")
        self.loaded_model_label.setStyleSheet("""
            QLabel {
                margin-bottom: 12px; 
                font-weight: 500;
                color: #333;
            }
        """)

        # Create buttons
        self.load_btn = QPushButton("📂 Load Model")
        self.test_btn = QPushButton("📊 Evaluate Model")

        # Style buttons
        button_style = """
            QPushButton {
                background-color: #041453;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 11pt;
                min-height: 36px;
            }
            QPushButton:hover {
                background-color: #1a3a5a;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """

        self.load_btn.setStyleSheet(button_style)
        self.test_btn.setStyleSheet(button_style.replace("#0f2740", "#7EB400").replace("#1a3a5a", "#8FC400"))

        # Connect button signals
        self.load_btn.clicked.connect(self.load_model)
        self.test_btn.clicked.connect(self.evaluate_model)

        # Disable evaluate button initially
        self.test_btn.setEnabled(False)

        # Add widgets to layout
        layout.addWidget(self.loaded_model_label)
        layout.addWidget(self.load_btn)
        layout.addWidget(self.test_btn)

        # Add analysis section
        analysis_title = QLabel("Performance Analysis")
        analysis_title.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 12pt;
                color: #041453;
                padding: 4px 0;
                margin-top: 20px;
            }
        """)
        layout.addWidget(analysis_title)

        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 8px;
                background: white;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10.5pt;
            }
        """)
        self.analysis_text.setPlaceholderText("Performance analysis will appear here after evaluation...")
        layout.addWidget(self.analysis_text)

        return layout

    def create_center_panel(self):
        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(0, 0, 0, 0)

        self.center_grid = QGridLayout()
        self.center_grid.setSpacing(12)
        self.center_grid.setContentsMargins(0, 0, 0, 0)

        viz_containers = [
            ("Loss vs Epochs", "loss_vs_epochs", 0, 0),
            ("Accuracy vs Epochs", "accuracy_vs_epochs", 0, 1),
            ("Feature Importance", "feature_importance", 0, 2),
            ("Confusion Matrix", "confusion_matrix", 1, 0),
            ("ROC Curve", "roc_curve", 1, 1),
            ("Metrics Table", "metrics_table", 1, 2)
        ]

        self.graph_labels = {}
        self.title_labels = {}

        for title, viz_key, row, col in viz_containers:
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setSpacing(4)
            container_layout.setContentsMargins(0, 0, 0, 0)

            title_label = QLabel(title)
            title_label.setStyleSheet("color: #041453")
            self.title_labels[viz_key] = title_label
            container_layout.addWidget(title_label)

            if viz_key == 'model_comparison':
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setStyleSheet("border: 1px solid #ccc; border-radius: 4px;")

                graph_label = QLabel("Metrics will appear here")
                graph_label.setAlignment(Qt.AlignCenter)
                graph_label.setStyleSheet("""
                    QLabel {
                        border: none;
                        background: white;
                    }
                """)
                scroll.setWidget(graph_label)
                container_layout.addWidget(scroll)
            else:
                graph_label = QLabel()
                graph_label.setAlignment(Qt.AlignCenter)
                graph_label.setStyleSheet("""
                    QLabel {
                        border: 1px solid #ccc;
                        border-radius: 4px;
                        background: white;
                    }
                """)
                container_layout.addWidget(graph_label)

            graph_label.setMinimumSize(300, 250)
            self.graph_labels[viz_key] = graph_label

            self.center_grid.addWidget(container, row, col)

        layout.addLayout(self.center_grid)

        # Comparison table at the bottom
        self.comparison_container = QWidget()
        comparison_layout = QVBoxLayout(self.comparison_container)



        self.comparison_title = QLabel("Model Comparison Table")
        self.comparison_title.setStyleSheet("color: #041453")
        comparison_layout.addWidget(self.comparison_title)



        comparison_scroll = QScrollArea()
        comparison_scroll.setWidgetResizable(True)
        comparison_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
             QWidget > QScrollArea {
                background: transparent;
            }
        """)

        self.comparison_label = QLabel("Loading model history...")
        self.comparison_label.setAlignment(Qt.AlignCenter)
        self.comparison_label.setTextFormat(Qt.RichText)
        self.comparison_label.setStyleSheet("""
            QLabel {
                padding: 0;
                margin: 0;
                background: white;
            }
        """)
        comparison_scroll.setWidget(self.comparison_label)

        comparison_layout.addWidget(comparison_scroll)
        layout.addWidget(self.comparison_container)

        return layout

    def create_right_panel(self):
        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(0, 0, 0, 0)

        training_group = QGroupBox()
        form_layout = QVBoxLayout()
        form_layout.setSpacing(12)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(100)
        self.epochs.setMinimumHeight(30)

        self.batch = QSpinBox()
        self.batch.setRange(1, 512)
        self.batch.setValue(32)
        self.batch.setMinimumHeight(30)

        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(3)
        self.lr.setRange(0.001, 1.0)
        self.lr.setValue(0.001)
        self.lr.setSingleStep(0.001)
        self.lr.setMinimumHeight(30)

        self.model_name = QLineEdit()
        self.model_name.setPlaceholderText("Enter model name")
        self.model_name.setMinimumHeight(30)

        self.train_btn = QPushButton("⚙️ Train Model")
        self.modern_button_style(self.train_btn, bg_color="#041453")
        self.train_btn.clicked.connect(self.start_training)

        form_layout.addWidget(QLabel("Epochs:"))
        form_layout.addWidget(self.epochs)
        form_layout.addWidget(QLabel("Batch size:"))
        form_layout.addWidget(self.batch)
        form_layout.addWidget(QLabel("Learning Rate:"))
        form_layout.addWidget(self.lr)
        form_layout.addWidget(QLabel("Model Name:"))
        form_layout.addWidget(self.model_name)
        form_layout.addStretch()
        form_layout.addWidget(self.train_btn)
        training_group.setLayout(form_layout)
        layout.addWidget(training_group)


        log_group = QGroupBox()
        log_layout = QVBoxLayout()

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMinimumHeight(200)
        self.train_log.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                color: #5a9e00;
            }
        """)

        log_layout.addWidget(self.train_log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        self.reset_btn = QPushButton("❌ Reset")
        self.modern_button_style(self.reset_btn, bg_color="#cc0000")
        self.reset_btn.setFixedHeight(36)
        self.reset_btn.clicked.connect(self.reset_fields)
        layout.addWidget(self.reset_btn, alignment=Qt.AlignRight)

        return layout

    def load_model(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "H5 Files (*.h5);;All Files (*)",
            options=options
        )

        if file_path:
            self.current_model_path = file_path.replace('\\', '/')
            self.loaded_model_label.setText(f"Loaded: {self.current_model_path.split('/')[-1]}")
            self.test_btn.setEnabled(True)
            self.update_log(f"Model loaded from: {self.current_model_path}")

    def evaluate_model(self):
        if not hasattr(self, 'current_model_path') or not self.current_model_path:
            QMessageBox.warning(self, "Error", "No model loaded!")
            return

        self.test_btn.setEnabled(False)
        self.test_btn.setText("Evaluating...")

        for label in self.graph_labels.values():
            if isinstance(label, QLabel):
                label.clear()
                label.setStyleSheet(self.viz_light_style)

        self.eval_thread = EvaluateModelThread(
            api_url=self.api_url,
            model_path=self.current_model_path
        )
        self.eval_thread.finished_signal.connect(self.on_evaluation_finished)
        self.eval_thread.error_signal.connect(self.on_evaluation_error)
        self.eval_thread.log_signal.connect(self.update_log)
        self.eval_thread.start()

    def on_evaluation_finished(self, result):
        self.test_btn.setEnabled(True)
        self.test_btn.setText("📊 Evaluate Model")

        display_text = ""


        if 'performance_report' in result:
            display_text += result['performance_report']

        self.analysis_text.setPlainText(display_text.strip())

        self.analysis_text.setReadOnly(True)
        self.analysis_text.setStyleSheet("""
            QTextEdit {
                color: #041453;
            }
        """)

        # Handle visualizations if present
        if 'visualizations' in result:
            self.update_visualizations(result['visualizations'])

        self.load_model_comparison()
        QMessageBox.information(self, "Success", "Evaluation completed successfully!")

    def on_evaluation_error(self, error_msg):
        self.test_btn.setEnabled(True)
        self.test_btn.setText("📊 Evaluate Model")
        self.update_log(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)

    def update_visualizations(self, visualizations):
        """Update all visualization displays with new data"""
        for viz_key, content in visualizations.items():
            if viz_key not in self.graph_labels:
                continue

            label = self.graph_labels[viz_key]

            try:
                if viz_key == 'metrics_table':
                    # Define the base style first
                    base_style = """
                    <style>
                        .metrics-table {
                            margin: 0 auto;
                            border-collapse: collapse;
                            font-family: Arial, sans-serif;
                            font-size: 11pt;
                            color: #333333;
                            font-style: normal;
                        }
                        .metrics-table th {
                            background-color: #f2f2f2;
                            padding: 8px;
                            text-align: center;
                            font-weight: bold;
                            border-bottom: 2px solid #ddd;
                        }
                        .metrics-table td {
                            padding: 8px;
                            border-bottom: 1px solid #ddd;
                            text-align: center;
                        }
                        .metrics-table tr:nth-child(even) {
                            background-color: #f9f9f9;
                        }
                        .metrics-value {
                            font-family: Arial, sans-serif;
                            color: #0066cc;
                        }
                    </style>
                    """

                    if isinstance(content, dict):
                        metrics = content
                        styled_content = f"""{base_style}
                        <table class="metrics-table">
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Accuracy</td>
                                <td class="metrics-value">{metrics.get('accuracy', 0):.4f}</td>
                            </tr>
                            <tr>
                                <td>Precision</td>
                                <td class="metrics-value">{metrics.get('precision', 0):.4f}</td>
                            </tr>
                            <tr>
                                <td>Recall</td>
                                <td class="metrics-value">{metrics.get('recall', 0):.4f}</td>
                            </tr>
                            <tr>
                                <td>F1 Score</td>
                                <td class="metrics-value">{metrics.get('f1', 0):.4f}</td>
                            </tr>
                        </table>
                        """
                        label.setText(styled_content)
                    else:
                        try:
                            # Handle case where content is HTML string
                            label.setText(base_style + content)
                        except Exception as e:
                            label.setText("⚠️ Error: Invalid metrics table format.")
                    label.setTextFormat(Qt.RichText)
                    label.setAlignment(Qt.AlignCenter)
                else:
                    # Handle image visualizations
                    if not content:
                        continue

                    img_data = b64decode(content)
                    pixmap = QPixmap()
                    if not pixmap.loadFromData(img_data):
                        raise ValueError("Failed to load image data")

                    # Store the original pixmap as property
                    label.setProperty("full_image", pixmap)

                    # Show scaled version in the main UI
                    scaled_pixmap = pixmap.scaled(
                        label.width(),
                        label.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    label.setPixmap(scaled_pixmap)

                    # Disconnect any existing mouse press event first
                    try:
                        label.mousePressEvent.disconnect()
                    except:
                        pass

                    # Connect the mouse press event
                    label.mousePressEvent = lambda event, lbl=label: self.show_enlarged_image(
                        lbl.property("full_image"))
                    label.setCursor(Qt.PointingHandCursor)
                    label.setToolTip("Click to enlarge")
                    label.setStyleSheet("""
                        QLabel {
                            border: 1px solid #ccc;
                            border-radius: 4px;
                            background: white;
                        }
                        QLabel:hover {
                            border: 2px solid #041453;
                        }
                    """)

            except Exception as e:
                logging.error(f"Error displaying {viz_key}: {str(e)}")
                label.setText(f"Error loading visualization: {str(e)}")

    def show_enlarged_image(self, pixmap):
        """Show an enlarged version of the image in a dialog"""
        if not isinstance(pixmap, QPixmap) or pixmap.isNull():
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Enlarged View")
        dialog.setModal(True)

        # Calculate size based on image and screen dimensions
        screen = QApplication.primaryScreen().availableGeometry()
        img_ratio = pixmap.width() / pixmap.height()

        # Set maximum dimensions (80% of screen size)
        max_width = int(screen.width() * 0.8)
        max_height = int(screen.height() * 0.8)

        # Calculate dimensions to maintain aspect ratio
        if img_ratio > 1:  # Landscape
            width = min(max_width, pixmap.width())
            height = int(width / img_ratio)
        else:  # Portrait or square
            height = min(max_height, pixmap.height())
            width = int(height * img_ratio)

        dialog.resize(width + 20, height + 60)  # Add space for borders/button

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setPixmap(pixmap.scaled(
            width,
            height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        close_btn.setFixedSize(100, 30)

        layout.addWidget(image_label, 1)
        layout.addWidget(close_btn, 0, alignment=Qt.AlignCenter)

        dialog.exec_()

    def start_training(self):
        params = {
            "epochs": self.epochs.value(),
            "batch_size": self.batch.value(),
            "learning_rate": self.lr.value(),
            "model_name": self.model_name.text().strip()
        }

        if not params["model_name"]:
            QMessageBox.warning(self, "Error", "Please enter a model name")
            return

        self.train_btn.setEnabled(False)
        self.train_btn.setText("Training...")
        self.train_log.clear()

        self.train_thread = TrainModelThread(
            api_url=f"{self.api_url}/train_and_save_model",
            params=params
        )
        self.train_thread.finished_signal.connect(self.on_training_finished)
        self.train_thread.error_signal.connect(self.on_training_error)
        self.train_thread.log_signal.connect(self.update_log)
        self.train_thread.start()

    def on_training_finished(self, result):
        self.train_btn.setEnabled(True)
        self.train_btn.setText("⚙️ Train Model")

        message = (
            f"\nTraining completed successfully!\n\n"
            f"Model: {result['message']}\n\n"
            f"Metrics:\n"
            f"Accuracy: {result['metrics']['accuracy']:.4f}\n"
            f"Precision: {result['metrics']['precision']:.4f}\n"
            f"Recall: {result['metrics']['recall']:.4f}\n"
            f"F1 Score: {result['metrics']['f1']:.4f}"
        )

        self.update_log(message)

        if 'visualizations' in result:
            self.update_visualizations(result['visualizations'])

        self.load_model_comparison()

        QMessageBox.information(self, "Success", message)

    def on_training_error(self, error_msg):
        self.train_btn.setEnabled(True)
        self.train_btn.setText("⚙️ Train Model")
        self.update_log(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)

    def update_log(self, message):
        self.train_log.append(message)
        self.train_log.verticalScrollBar().setValue(
            self.train_log.verticalScrollBar().maximum()
        )

    def reset_fields(self):
        self.epochs.setValue(100)
        self.batch.setValue(32)
        self.lr.setValue(0.001)
        self.model_name.clear()
        self.train_log.clear()

        if hasattr(self, 'current_model_path'):
            del self.current_model_path
        self.loaded_model_label.setText("Loaded: No model selected")
        self.test_btn.setEnabled(False)
        self.analysis_text.clear()

        for label in self.graph_labels.values():
            if isinstance(label, QLabel):
                label.clear()

                label.setStyleSheet(self.viz_light_style)


    def load_model_comparison(self):
        try:
            log_path = Path(MODEL_LOG_PATH) if isinstance(MODEL_LOG_PATH, str) else MODEL_LOG_PATH

            if not log_path.exists():
                self.comparison_label.setText("No model history available")
                return

            try:
                with open(log_path, mode='r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    models = list(reader)
            except UnicodeDecodeError:
                with open(log_path, mode='r', encoding='latin-1') as file:
                    reader = csv.DictReader(file)
                    models = list(reader)

            if not models:
                self.comparison_label.setText("No models in history")
                return

            html = """
            <style>
                .comparison-container {
                    display: block;
                    width: 100%;
                    background: transparent;
                    margin: 0;
                    padding: 0;
                }
                .comparison-table {
                    border-collapse: collapse;
                    font-family: Arial, sans-serif;
                    font-size: 11pt;
                    color: #333333;
                    width: 100%;
                    margin: 0;
                    padding: 0;
                }
                .comparison-table th {
                    background-color: #f2f2f2;
                    padding: 8px 12px;
                    text-align: center;
                    font-weight: bold;
                    border: 1px solid #ddd;
                }
                .comparison-table td {
                    padding: 8px 12px;
                    text-align: center;
                    border: 1px solid #ddd;
                    background-color: white;
                }
                .comparison-table tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .comparison-table tr:hover {
                    background-color: #f0f0f0;
                }
                .numeric-value {
                    font-family: Arial, sans-serif;
                    color: #0066cc;
                }
                .model-name {
                    max-width: 150px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                    color: #333333;
                    text-align: left;
                }
                .timestamp {
                    color: #666666;
                    font-size: 10pt;
                }
            </style>
            <div class="comparison-container">
                <table class="comparison-table">
                    <tr>
                        <th>Timestamp</th>
                        <th>Model Name</th>
                        <th>Accuracy</th>
                        <th>F1 Score</th>
                        <th>Epochs</th>
                        <th>Batch Size</th>
                        <th>Learning Rate</th>
                        <th>Training Time (s)</th>
                    </tr>
            """

            for model in sorted(models, key=lambda x: x['Timestamp'], reverse=True):
                html += f"""
                <tr>
                    <td class="timestamp">{model['Timestamp'][:16]}</td>
                    <td class="model-name" title="{model['Model Name']}">{model['Model Name'][:20]}{'...' if len(model['Model Name']) > 20 else ''}</td>
                    <td class="numeric-value">{float(model['Accuracy']):.3f}</td>
                    <td class="numeric-value">{float(model['F1-Score']):.3f}</td>
                    <td class="numeric-value">{model['Epochs']}</td>
                    <td class="numeric-value">{model['Batch Size']}</td>
                    <td class="numeric-value">{float(model['Learning Rate']):.3f}</td>
                    <td class="numeric-value">{float(model['Training Time (s)']):.1f}</td>
                </tr>
                """

            html += """
                </table>
            </div>
            """

            self.comparison_label.setText(html)
            self.comparison_label.setTextFormat(Qt.RichText)


        except Exception as e:
            error_msg = f"<p style='color:red'>Error loading comparison: {str(e)}</p>"
            self.comparison_label.setText(error_msg)
            self.comparison_label.setTextFormat(Qt.RichText)
            logging.error(f"Error loading model comparison: {str(e)}")

    def run_on_ui_thread(self, fn, *args):
        QTimer.singleShot(0, lambda: fn(*args))


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = OdorClassificationUI()
        window.showMaximized()
        ret = app.exec_()
        sys.exit(ret)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)