# Stroke Prediction ML Pipeline

## Overview
This project is a complete machine learning pipeline for predicting stroke risk using the **Kaggle Stroke Prediction Dataset**.  
It covers the entire process of data mining: from raw data loading and preprocessing, through exploratory data analysis (EDA), to training multiple machine learning classifiers and evaluating their performance.  
The project is designed to be modular, reproducible, and easy to extend with additional models or features.

---

## Dataset
- **Source**: [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)  
- **Features** include:
  - Demographics (gender, age, marital status, residence type, work type)
  - Health-related factors (hypertension, heart disease, smoking status, average glucose level, BMI)
- **Target variable**: `stroke` (0 = no stroke, 1 = stroke)

---

## Features
- **Data Preprocessing**  
  - Handling missing values (BMI, categorical features)  
  - Encoding categorical variables using One-Hot Encoding  
  - Standardizing numerical features  

- **Exploratory Data Analysis (EDA)**  
  - Target balance visualization  
  - Age distribution  
  - Glucose and BMI distributions compared by stroke outcome  

- **Model Training**  
  - Models included:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - Support Vector Classifier
    - K-Nearest Neighbors
  - GridSearchCV for hyperparameter optimization  
  - Cross-validation (StratifiedKFold)  
  - Class imbalance handling via class weights  

- **Evaluation**  
  - Classification reports (precision, recall, F1-score)  
  - ROC curves and AUC scores  
  - Confusion matrices  
  - Summary of best hyperparameters and metrics  

- **Outputs**  
  - Trained models saved as `.joblib` files  
  - Reports and plots (ROC, confusion matrix, classification reports)  
  - Feature names after preprocessing (JSON)  

---

## Project Structure

├─ data/ 
├─ outputs/
│ ├─ models/ # trained models
│ └─ reports/ # metrics, plots, and evaluation reports
├─ src/ # source code
│ ├─ config.py # global paths and constants
│ ├─ data.py # load and clean data
│ ├─ features.py # preprocessing and feature engineering
│ ├─ models.py # ML models and hyperparameters
│ ├─ train.py # training and saving models
│ ├─ evaluate.py # evaluation and metrics
│ ├─ visualize.py # quick EDA visualizations
│ ├─ utils.py # helper functions
│ └─ main.py # main entry point
├─ requirements.txt
└─ README.md


---

## Installation
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

Usage
Run full training (all models):
python -m src.main --csv data/stroke_data.csv

Run EDA only:
python -m src.main --csv data/stroke_data.csv --eda

Train specific models (example: Logistic Regression + Random Forest):
python -m src.main --csv data/stroke_data.csv --models logreg,rf

Outputs

Models: outputs/models/*.joblib

Reports:

classification_report.txt

confusion_matrix.png

roc_curve.png

feature_names.json

Future Improvements

Add deep learning models (e.g., neural networks with PyTorch/TensorFlow)

Apply SMOTE or other resampling techniques for better class balance

Deploy as a web app with Flask/FastAPI + React front-end

Integrate SHAP/feature importance explainability
