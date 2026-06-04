"""
Classification Algorithm Showdown
----------------------------------
A modular, clean, and well-documented Python project that implements a machine 
learning classification workflow using Scikit-Learn. It compares three models:
1. Logistic Regression (with L2 Regularization)
2. Support Vector Machine (RBF Kernel)
3. Random Forest Classifier

The script demonstrates:
- Synthetic missing value and categorical feature injection for demonstration.
- ColumnTransformer preprocessing pipelines (imputation, scaling, one-hot encoding).
- Prevention of data leakage via Pipeline orchestration.
- Stratified 5-Fold Cross-Validation.
- Learning curves for bias/variance diagnostics.
- Test-set metrics (Precision, Recall, F1, Confusion Matrix, and ROC/AUC curves).

Author: Antigravity (ML Engineer Agent)
Date: 2026-06-04
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_fscore_support
)

# Set high-resolution plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


def load_and_prepare_data(random_seed=42):
    """
    Loads the Breast Cancer Wisconsin dataset and simulates a realistic messy dataset
    by injecting missing values and a categorical column.
    
    ML Intuition:
    - Real-world data is rarely purely numeric or clean.
    - We inject ~5% missing values and a categorical feature ('clinical_stage') to
      demonstrate how standard preprocessing pipelines handle mixed data types and NaNs.
    """
    print("[1/6] Loading Breast Cancer Wisconsin dataset...")
    # Load dataset as a pandas DataFrame
    raw_data = load_breast_cancer(as_frame=True)
    df = raw_data.frame.copy()
    
    np.random.seed(random_seed)
    
    # 1. Synthetically inject missing values (NaNs) in key numeric columns
    # We choose 3 features and set 5% of their values to NaN
    corrupt_cols = ['mean radius', 'mean texture', 'mean area']
    print(f"      Injecting 5% missingness into numeric columns: {corrupt_cols}")
    for col in corrupt_cols:
        mask = np.random.rand(len(df)) < 0.05
        df.loc[mask, col] = np.nan
        
    # 2. Synthetically inject a categorical column 'clinical_stage'
    # We make the stage correlated with 'mean area' to mimic real clinical criteria,
    # adding noise so it isn't a perfect predictor.
    print("      Adding synthetic categorical feature 'clinical_stage'...")
    mean_area_filled = df['mean area'].fillna(df['mean area'].median())
    tertiles = np.percentile(mean_area_filled, [33.3, 66.6])
    
    stages = []
    for val in mean_area_filled:
        noise = np.random.rand()
        if noise < 0.15:  # 15% random assignment to simulate real-world noise
            stages.append(np.random.choice(['Stage I', 'Stage II', 'Stage III']))
        elif val < tertiles[0]:
            stages.append('Stage I')
        elif val < tertiles[1]:
            stages.append('Stage II')
        else:
            stages.append('Stage III')
            
    df['clinical_stage'] = stages
    
    # 3. Inject missing values into the categorical column as well
    mask_cat = np.random.rand(len(df)) < 0.05
    df.loc[mask_cat, 'clinical_stage'] = np.nan
    
    # Separate features and target
    # target: 0 = Malignant, 1 = Benign
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Perform Stratified Train-Test Split (80/20)
    # Stratification ensures that the train and test sets have the same proportion of
    # malignant vs benign cases, which is critical for imbalanced or smaller datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_seed
    )
    
    print(f"      Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")
    return X_train, X_test, y_train, y_test


def build_preprocessing_pipeline(numeric_cols, categorical_cols):
    """
    Builds a ColumnTransformer to preprocessing numeric and categorical columns.
    
    ML Intuition:
    - Numerical Imputation: We use 'median' instead of 'mean' because the median is robust 
      to outliers in physical measurements (e.g., cell area, perimeter).
    - Feature Scaling: Logistic Regression (gradient-descent based) and SVM (distance-based)
      are highly sensitive to the scale of features. StandardScaler centers features to 0-mean
      and unit variance. Random Forests are scale-invariant, but scaling does not hurt.
    - Categorical Imputation: We replace missing values with the 'most_frequent' category.
    - Categorical Encoding: OneHotEncoder converts nominal categories ('Stage I', 'Stage II', etc.)
      into numerical binary vectors. We set handle_unknown='ignore' so that if any new category 
      shows up in the test set, it gets encoded as all zeros instead of crashing.
    """
    print("[2/6] Designing preprocessing pipelines...")
    
    # Transformer pipeline for numeric columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Transformer pipeline for categorical columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Bundle preprocessing for both data types
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor


def create_model_pipelines(preprocessor, random_seed=42):
    """
    Wraps the estimators inside pipelines containing the shared preprocessor.
    
    ML Intuition:
    - Wrapping both preprocessing AND estimators in a single Scikit-Learn Pipeline is 
      the GOLD STANDARD for preventing DATA LEAKAGE.
    - During cross-validation, the scaling parameters (mean, variance) and imputation values (median, mode) 
      are calculated ONLY on the training folds and applied to validation folds, preventing test info 
      from bleeding into training.
    """
    pipelines = {
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                solver='lbfgs', max_iter=1000, random_state=random_seed
            ))
        ]),
        'Support Vector Machine': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', CalibratedClassifierCV(
                estimator=SVC(kernel='rbf', C=1.0, random_state=random_seed),
                ensemble=False
            ))
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=random_seed
            ))
        ])
    }
    return pipelines


def evaluate_baseline_cv(pipelines, X_train, y_train):
    """
    Evaluates baseline models using 5-Fold Stratified Cross-Validation on the training set.
    
    ML Intuition:
    - A simple train-validation split can have high variance depending on how data is partitioned.
    - Stratified K-Fold cross-validation splits the data into K parts while keeping class ratios constant.
      It trains on K-1 folds and validates on the remaining fold, rotating K times. 
    - This gives a robust estimate of generalization performance (mean) and model stability (standard deviation).
    """
    print("[3/6] Running Stratified 5-Fold Cross-Validation on training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    cv_results = {}
    for name, pipeline in pipelines.items():
        print(f"      Evaluating {name}...")
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            'Accuracy': (scores['test_accuracy'].mean(), scores['test_accuracy'].std()),
            'Precision': (scores['test_precision'].mean(), scores['test_precision'].std()),
            'Recall': (scores['test_recall'].mean(), scores['test_recall'].std()),
            'F1-Score': (scores['test_f1'].mean(), scores['test_f1'].std())
        }
        
    return cv_results


def plot_learning_curves(pipelines, X_train, y_train, filename='learning_curves.png'):
    """
    Generates and saves learning curves for all three models to diagnose under/overfitting.
    
    ML Intuition:
    - Learning curves plot training and cross-validation scores against varying training set sizes.
    - High Bias (Underfitting): Both training and validation curves plateau early at a low score.
      Adding more data will NOT help. The model is too simple (needs more features or complexity).
    - High Variance (Overfitting): There is a wide gap between training score (very high) and
      validation score (lower). Adding more training data, feature selection, or regularization will help.
    """
    print("[4/6] Computing and plotting learning curves...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 5)
    
    for i, (name, pipeline) in enumerate(pipelines.items()):
        # learning_curve automatically handles the fitting and score generation
        train_sizes_abs, train_scores, test_scores = learning_curve(
            pipeline, X_train, y_train, cv=cv, train_sizes=train_sizes,
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        ax = axes[i]
        
        # Plot training accuracy curve (Blue)
        ax.plot(train_sizes_abs, train_mean, 'o-', color='#1a73e8', label='Training Score', linewidth=2)
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#1a73e8')
        
        # Plot cross-validation accuracy curve (Orange)
        ax.plot(train_sizes_abs, test_mean, 's-', color='#e67e22', label='Validation Score (CV)', linewidth=2)
        ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.15, color='#e67e22')
        
        ax.set_title(f'{name}', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Training Samples', fontsize=11)
        if i == 0:
            ax.set_ylabel('Accuracy Score', fontsize=11)
        ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
        ax.set_ylim(0.75, 1.02)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.suptitle("Model Learning Curves (Bias vs. Variance Diagnosis)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {filename}")


def plot_evaluation_metrics(pipelines, X_train, y_train, X_test, y_test, 
                             roc_filename='roc_curves.png', 
                             cm_filename='confusion_matrices.png'):
    """
    Fits each model on the full training set, makes predictions on the test set,
    and creates overlapping ROC plots and Confusion Matrices.
    
    ML Intuition:
    - Confusion Matrix: Explodes predictions into True Positives, False Positives, 
      True Negatives, and False Negatives. Crucial in medicine (Breast Cancer), 
      where False Negatives (malignant tumors classified as benign) are far more dangerous 
      than False Positives.
    - Precision vs Recall:
      - Precision: Out of all predicted benign cases, how many were actually benign?
      - Recall: Out of all actual benign cases, how many did the model find?
    - ROC Curve & AUC: Plots True Positive Rate (Sensitivity) vs. False Positive Rate (1-Specificity)
      across all classification thresholds. AUC measures the model's ability to rank items
      (i.e., probability of a random positive sample having a higher predicted score than a random negative).
      AUC of 1.0 is perfect; 0.5 is random guessing.
    """
    print("[5/6] Training models on full train set and evaluating on test set...")
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))
    
    plt.figure(figsize=(10, 8))
    
    performance_reports = {}
    
    # Modern design color palette for models
    colors = {
        'Logistic Regression': '#1a73e8',     # Vibrant Blue
        'Support Vector Machine': '#9b59b6',    # Deep Purple
        'Random Forest': '#2ecc71'             # Emerald Green
    }
    
    for i, (name, pipeline) in enumerate(pipelines.items()):
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] # Probability of positive class (Benign)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        ax_cm = axes_cm[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                    annot_kws={"size": 16, "weight": "bold"})
        ax_cm.set_title(f'{name}', fontsize=14, fontweight='bold', pad=12)
        ax_cm.set_xlabel('Predicted Label', fontsize=11)
        ax_cm.set_ylabel('True Label', fontsize=11)
        ax_cm.set_xticklabels(['Malignant (0)', 'Benign (1)'], fontsize=10)
        ax_cm.set_yticklabels(['Malignant (0)', 'Benign (1)'], fontsize=10, rotation=0)
        ax_cm.grid(False)
        
        # 2. Performance Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        performance_reports[name] = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Classification Report': classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])
        }
        
        # 3. ROC and AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot overlapping ROC curves
        plt.plot(fpr, tpr, color=colors[name], lw=3, label=f'{name} (AUC = {roc_auc:.4f})')
        
    # Save Confusion Matrices figure
    fig_cm.suptitle("Confusion Matrices Showdown (Test Set)", fontsize=16, fontweight='bold', y=1.02)
    fig_cm.tight_layout()
    fig_cm.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.close(fig_cm)
    print(f"      Saved: {cm_filename}")
    
    # Finalize and Save ROC Curve figure
    plt.plot([0, 1], [0, 1], color='#7f8c8d', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.5000)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='semibold', labelpad=10)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='semibold', labelpad=10)
    plt.title('ROC Curve Comparison (Test Set)', fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(roc_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {roc_filename}")
    
    return performance_reports


def main():
    print("=" * 60)
    print("      CLASSIFICATION ALGORITHM SHOWDOWN RUNNER")
    print("=" * 60)
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_and_prepare_data(random_seed=42)
    
    # Separate numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # 2. Build Preprocessor
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    
    # 3. Create Models
    pipelines = create_model_pipelines(preprocessor, random_seed=42)
    
    # 4. Cross Validation
    cv_results = evaluate_baseline_cv(pipelines, X_train, y_train)
    
    # Print CV Results
    print("\n" + "=" * 60)
    print("      CROSS-VALIDATION PERFORMANCE (5-Fold Mean ± Std)")
    print("=" * 60)
    for name, metrics in cv_results.items():
        print(f"\nModel: {name}")
        for metric, (mean, std) in metrics.items():
            print(f"  - {metric:10}: {mean:.4f} ± {std:.4f}")
            
    # 5. Plot Learning Curves
    plot_learning_curves(pipelines, X_train, y_train)
    
    # 6. Evaluate on Test Set & Plot
    perf_reports = plot_evaluation_metrics(pipelines, X_train, y_train, X_test, y_test)
    
    # Print Test Set Reports
    print("\n" + "=" * 60)
    print("      TEST SET PERFORMANCE SHOWDOWN")
    print("=" * 60)
    
    # Construct a pretty summary table
    summary_data = []
    for name, reports in perf_reports.items():
        summary_data.append({
            'Model': name,
            'Precision': f"{reports['Precision']:.4f}",
            'Recall': f"{reports['Recall']:.4f}",
            'F1-Score': f"{reports['F1-Score']:.4f}"
        })
        print(f"\nDetailed Classification Report for {name}:")
        print(reports['Classification Report'])
        print("-" * 50)
        
    summary_df = pd.DataFrame(summary_data)
    print("\nPerformance Summary:")
    print(summary_df.to_markdown(index=False))
    print("\n" + "=" * 60)
    print(" Showdown complete. Visualizations saved to workspace.")
    print("=" * 60)


if __name__ == '__main__':
    main()
