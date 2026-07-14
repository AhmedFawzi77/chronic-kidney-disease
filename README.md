# Chronic Kidney Disease (CKD) Classification

A machine learning pipeline for predicting Chronic Kidney Disease from clinical measurements, developed as part of a Hospital Management System (HMS) graduation project.

## 📋 Overview

This project builds and evaluates multiple classification models to detect CKD from routine clinical test values, following a complete data science workflow: exploratory data analysis, preprocessing, model training with cross-validation, hyperparameter comparison, and model persistence for deployment.

## 📊 Dataset

- **Samples:** 400 patient records
- **Features:** 13 clinical measurements (e.g., blood pressure, specific gravity, albumin, sugar, blood urea, serum creatinine, sodium, potassium, hemoglobin, white/red blood cell counts, hypertension status)
- **Target:** Binary classification — CKD (1) vs. Not CKD (0)
- **Class distribution:** 250 CKD / 150 Not CKD

## 🔍 Methodology

1. **Exploratory Data Analysis** — class distribution, per-feature distributions by class, boxplots, correlation heatmap, and feature-target correlation analysis
2. **Preprocessing** — stratified train/test split (80/20), feature scaling with `StandardScaler`, class imbalance handled via `class_weight='balanced'`
3. **Model Training** — three models trained inside `scikit-learn` Pipelines:
   - Logistic Regression
   - Random Forest
   - XGBoost
4. **Validation** — 5-fold Stratified Cross-Validation (F1-scored) before final evaluation on the held-out test set
5. **Evaluation** — Accuracy, Precision, Recall, F1-score, AUC, confusion matrices, ROC curves, and Precision-Recall curves for all models
6. **Model Interpretation** — feature importance (Random Forest) and learning curves to check for overfitting/underfitting
7. **Model Selection & Export** — best-performing model automatically selected by F1-score and saved with `joblib`

## 📈 Results

**5-Fold Cross-Validation (F1-score):**

| Model | F1 (mean ± std) |
|---|---|
| Random Forest | 0.988 ± 0.011 |
| Logistic Regression | 0.982 ± 0.020 |
| XGBoost | 0.980 ± 0.013 |

**Test Set Performance:**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **Random Forest** | 1.000 | 1.00 | 1.00 | 1.000 | 1.000 |
| XGBoost | 0.975 | 0.98 | 0.98 | 0.980 | 0.999 |
| Logistic Regression | 0.975 | 1.00 | 0.96 | 0.980 | 0.997 |

**Best model:** Random Forest (saved as `best_ckd_model.pkl`)

## 🛠️ Tech Stack

- Python (pandas, numpy)
- scikit-learn (Pipeline, StandardScaler, LogisticRegression, RandomForestClassifier, model_selection, metrics)
- XGBoost
- matplotlib, seaborn
- joblib

## ▶️ How to Run

1. Clone the repository
2. Place `chronic kidney disease.csv` in the project root
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
   ```
4. Run the notebook `CKD_Classification.ipynb` cell by cell

## 📁 Project Context

This model is one of two disease-prediction models (alongside a Liver/Pancreatic Disease model) built for a Hospital Management System graduation project, where the trained models are served through a FastAPI backend for real-time prediction.

## 👤 Author

Ahmed Fawzi — [GitHub](https://github.com/AhmedFawzi77)
