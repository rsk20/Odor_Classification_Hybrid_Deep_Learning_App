 <img width="50" height="50" alt="logo_odor_app" src="https://github.com/user-attachments/assets/66d5dd24-451d-4319-a621-bf548b85c7b3" />
 
# Odor Classification Desktop App with Hybrid Deep Learning
### MSc Thesis Project – Epoka University  
**Title:** Enhancing Odor Classification in E-Nose Systems Using Advanced Deep Learning Techniques (2025)

---

## 📖 Overview

This project implements a **hybrid deep learning model** that combines **Convolutional Neural Networks (CNNs)**, **Long Short-Term Memory Networks (LSTMs)**, and **Transformer-based attention mechanisms** for **odor classification using e-nose sensor data**.  



It provides:

- **Backend (FastAPI)**  
  - Endpoints for training models with custom hyperparameters  
  - Model evaluation & prediction  
  - Visualization outputs (loss curves, accuracy, confusion matrix, ROC, feature importance)  
  - Model comparison logging  

- **Desktop Application (PyQt5/Dash)**  
  - User-friendly interface for training & evaluating models  
  - Load and save `.h5` models  
  - View training logs, graphs, metrics, and comparison tables  

The project uses the **Gas Sensor Array Drift Dataset (UCI)** as the benchmark dataset.

---

## 📂 Project Structure
```
Odor_Classification_Hybrid_Deep_Learning_App/
│── backend/               # FastAPI backend: training, evaluation, endpoints
│── data/                  # Dataset (Gas Sensor Array Drift), processed .npy/.xlsx files
│── models/                # Saved .h5 trained models
│── logs/                  # Training & error logs
│── visualizations/        # Plots (loss, accuracy, confusion matrix, ROC)
│── app_desktop.py         # PyQt5/Dash desktop application
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation (this file)
```

---

## ⚙️ Installation

### Runned and Tested through:

JetBrains IDE PyCharm 2023.3.4
Device Victus by HP Gaming Laptop
OS Windows 11
Processor	13th Gen Intel(R) Core(TM) i5-13420H (2.10 GHz)
System type	64-bit operating system, x64-based processor


### Used: 
- Python 3.11 or 3.12
- numpy 1.26.4
- pandas 2.2.1
- scikit-learn 1.6.1
- tensorflow 2.19.0
- plotly 6.0.1
- fastapi 0.115.12
- uvicorn 0.34.2
- matplotlib 3.8.3
- pydantic 2.10.6
- python-docx 0.8.11
- reportlab 4.4.0
- PyQt5 5.15.9
- httpx  0.13.3


### 1. Clone the Repository and Install Dependencies
```bash
git clone https://github.com/rsk20/Odor_Classification_Hybrid_Deep_Learning_App.git
cd Odor_Classification_Hybrid_Deep_Learning_App

pip install -r requirements.txt

```

## Dataset

The Gas Sensor Array Drift Dataset is required.

Preprocessed `.npy` and `.xlsx` versions should be placed in the data/ folder.

If missing, download from [UCI Machine Learning Repository.](https://archive.ics.uci.edu/ml/datasets/Gas+sensor+array+drift+dataset)

## 🚀 Running the Application
### Run Backend (FastAPI) first, then Frontend (Desktop App)

Start the API server:
```bash
python -m uvicorn backend.api:app --reload
```
Endpoints:
- POST `/train_and_save_model` → Train with parameters (epochs, batch size, learning rate)
- POST `/load_model_and_evaluate` → Load saved .h5 model and evaluate
- POST `/generate_analysis` → Generate metrics, graphs, and comparison logs

The Desktop App opens.

## Features:

- Load Model → Import `.h5` model

- Test Prediction → Evaluate on test data

- Train Model → Train new model with given hyperparameters

- Save Model → Save trained .h5 model

- Reset → Clear session and restart app

## 📊 Outputs & Visualizations

- Data Visualization Table

- Feature Importance Bar Chart

- Training Curves (Loss vs Epochs, Accuracy vs Epochs)

- Validation vs Training Performance

- Confusion Matrix & ROC Curve

- Metrics Table (Accuracy, Precision, Recall, F1)

- Model Comparison Table (with hyperparameters, accuracy, timestamp, ID)

<img width="2048" height="1080" alt="image" src="https://github.com/user-attachments/assets/d6b0ae5a-9c82-438c-b516-ad32840e64ff" />

📜 License 

This project is part of the MSc thesis of the author (@rsk20) at Epoka University (2025).
Usage of this code requires prior permission from the author.
For further information please read the LICENSE.md on https://github.com/rsk20/Odor_Classification_Hybrid_Deep_Learning_App/LICENSE.md
