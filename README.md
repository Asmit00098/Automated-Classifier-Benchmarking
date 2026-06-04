# ⚔️ Classification Algorithm Showdown & Dashboard

A modular, clean, and interactive machine learning project that compares three distinct classifiers (Logistic Regression, Support Vector Machine, and Random Forest) using Scikit-Learn. The project contains a command-line script for static evaluations and a Streamlit Web Dashboard for interactive datasets, custom csv uploads, live hyperparameter tuning, and real-time predictions.

---

## 📂 Project Structure

```bash
ML_Mini_project/
├── .venv/                      # Isolated Python Virtual Environment
├── app.py                      # Interactive Streamlit Web Dashboard
├── classification_showdown.py  # Production-grade ML Pipeline CLI Runner
└── README.md                   # Project documentation
```

---

## ✨ Features

1. **Robust Preprocessing Pipeline**: Handles both numerical columns (with outlier-resistant median imputation and z-score normalization scaling) and categorical columns (with mode imputation and one-hot encoding).
2. **Preventing Data Leakage**: Utilizes Scikit-Learn `Pipeline` and `ColumnTransformer` to ensure preprocessing statistics are calculated strictly inside training folds during cross-validation.
3. **Stratified 5-Fold Cross-Validation**: Assesses baseline classifier generalization and performance stability.
4. **Learning Curve Diagnostics**: Plots training vs. validation scores across varying training set sizes to diagnose underfitting (bias) and overfitting (variance).
5. **Metrics Showdown**: Generates side-by-side Confusion Matrices, ROC curves with area under the curve (AUC) metrics, precision, recall, and F1-Scores on a held-out test set.
6. **Dynamic Dashboard App**: An interactive web browser interface hosting all of the above, plus supporting **custom CSV dataset uploads**, **interactive hyperparameter sliders**, and a **real-time prediction generator**.

---

## 🧠 Machine Learning Design Choices & Intuition

### 1. Robust Imputation & Scaling
* **Median Imputation**: Physical observations (like cell area, texture, or financial details) often contain heavy tails or outliers. Imputing missing values with the median is robust to outliers, unlike mean imputation.
* **Standard Scaling**: Algorithms like Logistic Regression (gradient-descent driven) and SVM (distance/margin-driven) are scale-sensitive. If features are on different scales (e.g. Area vs. Radius), the larger scale features will dominate. Scaling standardizes them (mean = 0, variance = 1). Random Forest is scale-invariant but scaling does not affect its tree splits.

### 2. Preventing Data Leakage via Pipelines
* When performing cross-validation, scaling the entire dataset beforehand is a **critical mistake** (leaking test data mean/variance into training).
* By encapsulating preprocessing steps and model estimators inside a Scikit-Learn `Pipeline`, scaling and imputation parameters are computed solely on the training splits of each fold and applied to the validation splits, ensuring zero leakage.

### 3. Calibration for Support Vector Machines
* Standard SVM classifiers project data to high-dimensional spaces to find separating hyperplanes, but they do not naturally output probability scores.
* Rather than using the deprecated `probability=True` keyword in `SVC` (which relies on internal Platt scaling and is deprecated in Scikit-Learn 1.9+), this project wraps `SVC` inside `CalibratedClassifierCV` to compute calibrated class probabilities cleanly.

### 4. Diagnosing Curves
* **Learning Curves**: By graphing validation and training performance against increasing training sizes, we check for:
  * **High Bias (Underfitting)**: Curves converge early at a low accuracy.
  * **High Variance (Overfitting)**: A wide gap remains between training and validation accuracy.

---

## 🚀 Setup & Installation

Ensure you have Python 3.10+ installed.

### 1. Initialize Virtual Environment & Activate
Create and activate an isolated virtual environment:

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\activate
```

```bash
# macOS / Linux / Git Bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
Install requirements in your activated virtual environment:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn streamlit tabulate
```

---

## 💻 Usage

### A. Run CLI Pipeline Runner
Execute the script to load the messy Breast Cancer dataset, run Stratified 5-Fold CV, compute learning/ROC curves, and print a markdown comparison table to your terminal:

```bash
python classification_showdown.py
```

### B. Launch Streamlit Web Dashboard
Launch the interactive web application:

```bash
streamlit run app.py
```

After starting up, open your browser and navigate to:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 📈 Metric Showdown Results (Default Dataset)

| Model                  |   Precision |   Recall |   F1-Score |   ROC AUC |
|:-----------------------|------------:|---------:|-----------:|----------:|
| **Logistic Regression**|      0.9861 |   0.9861 |     0.9861 |    0.9983 |
| **Support Vector Machine**|   0.9859 |   0.9722 |     0.9790 |    0.9950 |
| **Random Forest**      |      0.9583 |   0.9583 |     0.9583 |    0.9917 |
