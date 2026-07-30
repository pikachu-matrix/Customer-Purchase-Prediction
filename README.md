# Customer Purchase Prediction System

A production-ready Machine Learning project that predicts whether a customer is likely to make a purchase based on historical customer and transaction data.

---

## Project Overview

This project demonstrates a complete end-to-end Machine Learning pipeline following software engineering best practices.

The pipeline includes:

- Data Loading
- Data Validation
- Data Preprocessing
- Data Quality Checks
- Feature Engineering
- Feature Selection
- Model Training
- Model Evaluation
- Model Comparison
- Model Saving
- Prediction on New Data
- Feature Importance Visualization

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

## Project Structure

```
Customer_Purchase_Prediction/
│
├── config/
│   └── config.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── outputs/
│   ├── reports/
│   ├── predictions/
│   └── feature_importance/
│
├── logs/
│
├── src/
│   ├── common/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── pipeline.py
│
├── requirements.txt
├── README.md
└── main.py
```

---

## Machine Learning Pipeline

### Training Pipeline

```
Load Data
      ↓
Validate Dataset
      ↓
Preprocess Data
      ↓
Data Quality Check
      ↓
Time Features
      ↓
Customer Features
      ↓
Gap Features
      ↓
Target Creation
      ↓
Feature Selection
      ↓
Train/Test Split
      ↓
Train Multiple Models
      ↓
Model Comparison
      ↓
Best Model Selection
      ↓
Save Model
      ↓
Feature Importance Report
```

---

### Prediction Pipeline

```
Load Test Data
      ↓
Validate Dataset
      ↓
Preprocess Data
      ↓
Data Quality Check
      ↓
Time Features
      ↓
Customer Features
      ↓
Gap Features
      ↓
Feature Selection
      ↓
Load Saved Model
      ↓
Generate Predictions
      ↓
Save Prediction File
```

---

## Models Used

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The model with the lowest Mean Squared Error (MSE) is automatically selected and saved.

---

## Features Engineered

The project creates several engineered features, including:

- Time-based Features
- Customer-level Features
- Purchase Gap Features
- Target Variable
- Selected Training Features

---

## Outputs Generated

The project generates:

- Trained Model
- Prediction CSV
- Feature Importance CSV
- Feature Importance Plot
- Training Logs

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd Customer_Purchase_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute:

```bash
python main.py
```

Choose:

```
1 → Train Model

2 → Predict using Saved Model
```

---

## Logging

The application uses a centralized logging system to record:

- Training Progress
- Prediction Progress
- Model Selection
- Errors
- Exceptions

---

## Exception Handling

The project implements a centralized custom exception framework for consistent error reporting across all modules.

---

## Future Improvements

- Hyperparameter Tuning
- Cross Validation
- Model Explainability (SHAP)
- REST API using FastAPI
- Docker Support
- CI/CD Pipeline
- Cloud Deployment
- MLflow Integration

---

## Author

Developed as a production-style Machine Learning project for learning and portfolio purposes.