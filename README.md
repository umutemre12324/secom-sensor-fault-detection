# SECOM Sensor Fault Detection

A deep learning project for detecting and classifying faulty semiconductor manufacturing processes using the SECOM dataset and a PyTorch-based LSTM-FCN model.

## Overview

Manufacturing processes rely on numerous sensors to monitor production quality. Identifying faulty processes is essential for reducing production losses and improving reliability.

This project uses the **SECOM dataset** to develop a binary classification model that distinguishes between normal and faulty manufacturing samples. The model combines Long Short-Term Memory (LSTM) and Fully Convolutional Network (FCN) components to learn feature representations from sensor measurements.

## Features

- **Data Preprocessing:** Handles missing values and removes features with excessive missing data.
- **Feature Scaling:** Applies `StandardScaler` to standardize sensor measurements.
- **Class Imbalance Handling:** Uses SMOTE on the training data to address class imbalance.
- **Deep Learning:** Combines LSTM and CNN layers in an LSTM-FCN architecture.
- **Model Training:** Trains the model using PyTorch and the Adam optimizer.
- **Model Evaluation:** Evaluates predictions using accuracy, a confusion matrix, and a classification report.

## Technologies

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn

## Dataset

This project uses the **SECOM dataset**, which contains sensor measurements collected during a semiconductor manufacturing process.

The dataset includes:

- Sensor measurements representing manufacturing process features.
- Binary pass/fail labels indicating whether a sample passed quality inspection.

The dataset is highly imbalanced and contains missing values, making preprocessing and appropriate evaluation important.

Dataset source: [UCI Machine Learning Repository — SECOM](https://archive.ics.uci.edu/dataset/179/secom)

Place the dataset files in the project root directory:

```text
secom.data
secom_labels.data
```

## Project Structure

```text
secom-sensor-fault-detection/
├── preprocessing.py
├── model.py
├── train.py
├── test.py
├── secom.data
├── secom_labels.data
├── scaler.pkl
└── README.md
```

`scaler.pkl` is generated during preprocessing. The trained model file `lstmfcn_model.pth` is generated after training.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/umutemre12324/secom-sensor-fault-detection.git
cd secom-sensor-fault-detection
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, activate the environment using Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install torch pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib
```

## Usage

Run the scripts in the following order.

### 1. Preprocess the dataset

```bash
python preprocessing.py
```

This step prepares the sensor data, handles missing values, scales the features, addresses class imbalance in the training set, and creates the data loaders.

### 2. Train the model

```bash
python train.py
```

The training script trains the LSTM-FCN model and saves the trained weights to:

```text
lstmfcn_model.pth
```

### 3. Evaluate the model

```bash
python test.py
```

The evaluation script loads the trained model and reports its performance on the test set.

## Model Architecture

The LSTM-FCN architecture combines two feature extraction branches:

1. **LSTM Branch:** Processes the sensor features and learns sequential representations.
2. **CNN Branch:** Uses one-dimensional convolutional layers to extract local feature patterns.
3. **Feature Fusion:** Concatenates the features from both branches.
4. **Classification Layer:** Produces predictions for the two classes: Normal and Faulty.

The model uses cross-entropy loss and the Adam optimizer.

**Note:** The SECOM dataset contains sensor measurements rather than a naturally ordered time series. Treating the feature columns as a sequence is a modeling assumption in this implementation.

## Evaluation Metrics

The model is evaluated using:

- **Accuracy:** Overall proportion of correct predictions.
- **Confusion Matrix:** Distribution of correct and incorrect predictions for each class.
- **Precision:** Reliability of positive fault predictions.
- **Recall:** Proportion of actual faulty samples correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.

Because the SECOM dataset is highly imbalanced, accuracy alone may not provide a complete picture of model performance. Recall and F1-score for the faulty class are especially important.

## Future Improvements

- Experiment with alternative feature selection methods.
- Compare LSTM-FCN with traditional machine learning classifiers.
- Tune hyperparameters to improve fault detection.
- Evaluate additional imbalance-handling strategies.
- Investigate explainability methods to better understand sensor-related predictions.

## Author

**Umut Emre Özdamar**

GitHub: [@umutemre12324](https://github.com/umutemre12324)

---

*This project was developed for educational purposes to explore deep learning techniques for industrial sensor fault detection.*
