

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

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

# ---------------------------------------------------------
# Page Configurations & Design Aesthetics
# ---------------------------------------------------------
st.set_page_config(
    page_title="Algorithm Showdown Dashboard",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design aesthetics (fonts, colors, cards)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Styling */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Titles & Subtitles */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1a73e8 0%, #9b59b6 50%, #2ecc71 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        padding-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.15rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    
    /* Styled Metric Card */
    .metric-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #eaeaea;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #7f8c8d;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Data Loaders & Preprocessing Functions
# ---------------------------------------------------------
@st.cache_data
def get_default_data(num_missing_rate, cat_missing_rate, seed=42):
    """Loads and caches the default Breast Cancer dataset with customizable corruption rates."""
    raw_data = load_breast_cancer(as_frame=True)
    df = raw_data.frame.copy()
    
    np.random.seed(seed)
    
    # Inject missing values in numerical features
    corrupt_cols = ['mean radius', 'mean texture', 'mean area']
    for col in corrupt_cols:
        if num_missing_rate > 0.0:
            mask = np.random.rand(len(df)) < num_missing_rate
            df.loc[mask, col] = np.nan
            
    # Inject synthetic categorical feature
    mean_area_filled = df['mean area'].fillna(df['mean area'].median())
    tertiles = np.percentile(mean_area_filled, [33.3, 66.6])
    
    stages = []
    for val in mean_area_filled:
        noise = np.random.rand()
        if noise < 0.15:
            stages.append(np.random.choice(['Stage I', 'Stage II', 'Stage III']))
        elif val < tertiles[0]:
            stages.append('Stage I')
        elif val < tertiles[1]:
            stages.append('Stage II')
        else:
            stages.append('Stage III')
            
    df['clinical_stage'] = stages
    
    # Inject missing values in categorical feature
    if cat_missing_rate > 0.0:
        mask_cat = np.random.rand(len(df)) < cat_missing_rate
        df.loc[mask_cat, 'clinical_stage'] = np.nan
        
    return df


def detect_columns(df, target_col):
    """Automatically detects numeric and categorical columns excluding the target."""
    features_df = df.drop(columns=[target_col])
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def build_pipeline(numeric_cols, categorical_cols, model):
    """Creates a unified Pipeline combining preprocessing and estimator."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Handles categorical features, ignoring unknown categories at test time
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline


# ---------------------------------------------------------
# Sidebar Panel: Configurations & Inputs
# ---------------------------------------------------------
st.sidebar.markdown("## ⚙️ Project Settings")

# 1. Dataset Selection
data_source = st.sidebar.selectbox(
    "Select Dataset Source",
    ["Default: Breast Cancer (Mixed & Messy)", "Upload Custom CSV"]
)

df = None
target_col = None

if data_source == "Default: Breast Cancer (Mixed & Messy)":
    # Adjustable corruption parameters for Breast Cancer
    st.sidebar.subheader("Simulate Data Issues")
    num_missing = st.sidebar.slider("Numerical Missingness Rate", 0.0, 0.25, 0.05, step=0.01)
    cat_missing = st.sidebar.slider("Categorical Missingness Rate", 0.0, 0.25, 0.05, step=0.01)
    df = get_default_data(num_missing, cat_missing)
    target_col = 'target'
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV Uploaded Successfully!")
            
            # Select target variable from columns
            all_cols = df.columns.tolist()
            target_col = st.sidebar.selectbox("Select Target Column (Y)", all_cols, index=len(all_cols)-1)
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
    else:
        st.sidebar.warning("Please upload a CSV file to proceed.")

# 2. Hyperparameter Settings (Expanders in Sidebar)
st.sidebar.markdown("## 🧠 Model Hyperparameters")

# Logistic Regression Parameters
with st.sidebar.expander("1. Logistic Regression"):
    lr_c = st.slider("C (Inverse Regularization)", 0.01, 10.0, 1.0, step=0.05, key="lr_c")
    lr_solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"], key="lr_solver")
    lr_max_iter = st.number_input("Max Iterations", 100, 5000, 1000, step=100, key="lr_max_iter")

# Support Vector Machine Parameters
with st.sidebar.expander("2. Support Vector Machine"):
    svm_c = st.slider("C (Regularization Cost)", 0.01, 10.0, 1.0, step=0.05, key="svm_c")
    svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="svm_kernel")
    svm_gamma = st.selectbox("Gamma", ["scale", "auto"], key="svm_gamma")

# Random Forest Parameters
with st.sidebar.expander("3. Random Forest"):
    rf_estimators = st.slider("N Estimators (Trees)", 10, 500, 100, step=10, key="rf_est")
    rf_depth = st.slider("Max Depth", 2, 20, 8, step=1, key="rf_depth")
    rf_split = st.slider("Min Samples Split", 2, 10, 2, step=1, key="rf_split")

# Training trigger button
run_training = st.sidebar.button("🚀 Train & Run Showdown", width="stretch")


# ---------------------------------------------------------
# Main Panel Layout & Tabs
# ---------------------------------------------------------
st.markdown("<div class='main-title'>Algorithm Showdown Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Compare classical classifiers, explore preprocessing, tune parameters, and analyze model diagnostics.</div>", unsafe_allow_html=True)

if df is not None and target_col is not None:
    # Set up columns list
    numeric_cols, categorical_cols = detect_columns(df, target_col)
    
    # Establish train-test split
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Try stratified split if target is discrete; fallback to normal split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except Exception:
        # Fallback if target can't be stratified
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Instantiate the models based on custom sidebar parameters
    models = {
        'Logistic Regression': LogisticRegression(
            C=lr_c, solver=lr_solver, max_iter=lr_max_iter, random_state=42
        ),
        'Support Vector Machine': CalibratedClassifierCV(
            estimator=SVC(C=svm_c, kernel=svm_kernel, gamma=svm_gamma, random_state=42),
            ensemble=False
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=rf_estimators, max_depth=rf_depth, 
            min_samples_split=rf_split, random_state=42
        )
    }

    # Build the full pipelines
    pipelines = {
        name: build_pipeline(numeric_cols, categorical_cols, model)
        for name, model in models.items()
    }

    # Trigger training and evaluation if clicked, or retrieve from Session State
    if run_training or 'cv_results' not in st.session_state:
        with st.spinner("🔄 Preprocessing data, running 5-Fold CV, and fitting test set..."):
            # 1. Stratified 5-Fold CV
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = ['accuracy', 'precision', 'recall', 'f1']
            
            cv_results = {}
            test_results = {}
            fitted_pipelines = {}
            
            for name, pipeline in pipelines.items():
                # Baseline CV
                try:
                    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
                    cv_results[name] = {
                        'Accuracy': scores['test_accuracy'].mean(),
                        'Accuracy_std': scores['test_accuracy'].std(),
                        'Precision': scores['test_precision'].mean(),
                        'Precision_std': scores['test_precision'].std(),
                        'Recall': scores['test_recall'].mean(),
                        'Recall_std': scores['test_recall'].std(),
                        'F1-Score': scores['test_f1'].mean(),
                        'F1_std': scores['test_f1'].std()
                    }
                except Exception as e:
                    # Fallback scoring if binary metrics fail (e.g. multiclass)
                    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
                    cv_results[name] = {
                        'Accuracy': scores['test_accuracy'].mean(),
                        'Accuracy_std': scores['test_accuracy'].std(),
                        'Precision': 0.0, 'Precision_std': 0.0,
                        'Recall': 0.0, 'Recall_std': 0.0,
                        'F1-Score': 0.0, 'F1_std': 0.0
                    }
                
                # Fit on full training set
                pipeline.fit(X_train, y_train)
                fitted_pipelines[name] = pipeline
                
                # Predict on test set
                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
                
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                cm = confusion_matrix(y_test, y_pred)
                
                # ROC Curve Calculation
                fpr, tpr, roc_auc = None, None, None
                if y_prob is not None:
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                
                test_results[name] = {
                    'y_pred': y_pred,
                    'y_prob': y_prob,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'ConfusionMatrix': cm,
                    'FPR': fpr,
                    'TPR': tpr,
                    'AUC': roc_auc,
                    'Report': classification_report(y_test, y_pred, output_dict=True)
                }
            
            # Save all run results in session state to preserve state between tab switches
            st.session_state['cv_results'] = cv_results
            st.session_state['test_results'] = test_results
            st.session_state['fitted_pipelines'] = fitted_pipelines
            st.session_state['data_hash'] = hash(df.values.tobytes())
            st.sidebar.success("Model Showdown Complete!")

    # Load from session state
    cv_results = st.session_state['cv_results']
    test_results = st.session_state['test_results']
    fitted_pipelines = st.session_state['fitted_pipelines']

    # Create Tabs
    tab_data, tab_cv, tab_learning, tab_test, tab_predict = st.tabs([
        "📂 Data Explorer", 
        "⚙️ Baseline CV Metrics", 
        "📈 Learning Curves", 
        "⚔️ Performance Showdown",
        "🔮 Live Prediction Tool"
    ])

    # ---------------------------------------------------------
    # TAB 1: DATA EXPLORER
    # ---------------------------------------------------------
    with tab_data:
        st.subheader("Data Overview & Statistics")
        
        col_shape1, col_shape2, col_shape3 = st.columns(3)
        with col_shape1:
            st.markdown(f"<div class='metric-container'><div class='metric-label'>Row Count</div><div class='metric-value'>{df.shape[0]}</div></div>", unsafe_allow_html=True)
        with col_shape2:
            st.markdown(f"<div class='metric-container'><div class='metric-label'>Feature Count</div><div class='metric-value'>{df.shape[1] - 1}</div></div>", unsafe_allow_html=True)
        with col_shape3:
            st.markdown(f"<div class='metric-container'><div class='metric-label'>Missing Values</div><div class='metric-value'>{df.isnull().sum().sum()}</div></div>", unsafe_allow_html=True)
            
        col_exp1, col_exp2 = st.columns([3, 2])
        with col_exp1:
            st.write("**Dataset Preview:**")
            st.dataframe(df.head(100), width="stretch", height=350)
            
        with col_exp2:
            st.write("**Feature Type breakdown:**")
            st.write(f"- **Numerical Columns ({len(numeric_cols)}):** `{', '.join(numeric_cols[:6])}`" + ("..." if len(numeric_cols) > 6 else ""))
            st.write(f"- **Categorical Columns ({len(categorical_cols)}):** `{', '.join(categorical_cols) if categorical_cols else 'None'}`")
            
            # Target class distribution visualization
            st.write("**Target Class Balance:**")
            class_counts = df[target_col].value_counts()
            fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
            sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette="Blues_d", legend=False, ax=ax_bar)
            ax_bar.set_ylabel("Count")
            ax_bar.set_xlabel("Target Label")
            fig_bar.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

    # ---------------------------------------------------------
    # TAB 2: BASELINE CV METRICS
    # ---------------------------------------------------------
    with tab_cv:
        st.subheader("5-Fold Cross-Validation Performance")
        st.write("Evaluating models across validation folds protects against optimistic bias on the training set.")
        
        cv_df_data = []
        for name, metrics in cv_results.items():
            cv_df_data.append({
                'Model': name,
                'Accuracy (Mean)': f"{metrics['Accuracy']:.4f} ± {metrics['Accuracy_std']:.4f}",
                'Precision (Mean)': f"{metrics['Precision']:.4f} ± {metrics['Precision_std']:.4f}",
                'Recall (Mean)': f"{metrics['Recall']:.4f} ± {metrics['Recall_std']:.4f}",
                'F1-Score (Mean)': f"{metrics['F1-Score']:.4f} ± {metrics['F1_std']:.4f}",
            })
        
        st.table(pd.DataFrame(cv_df_data))
        
        # Plot comparative bar chart of F1-Score & Accuracy
        st.write("**Comparative Performance Plot:**")
        fig_cv_bar, ax_cv_bar = plt.subplots(figsize=(10, 4))
        models_list = list(cv_results.keys())
        acc_means = [cv_results[m]['Accuracy'] for m in models_list]
        f1_means = [cv_results[m]['F1-Score'] for m in models_list]
        
        x = np.arange(len(models_list))
        width = 0.35
        
        ax_cv_bar.bar(x - width/2, acc_means, width, label='CV Accuracy', color='#1a73e8')
        ax_cv_bar.bar(x + width/2, f1_means, width, label='CV F1-Score', color='#2ecc71')
        
        ax_cv_bar.set_ylabel('Score')
        ax_cv_bar.set_title('Cross-Validation Score Comparison')
        ax_cv_bar.set_xticks(x)
        ax_cv_bar.set_xticklabels(models_list)
        ax_cv_bar.legend(loc='lower right')
        ax_cv_bar.set_ylim(0.7, 1.02)
        ax_cv_bar.grid(True, linestyle=':', alpha=0.6)
        
        st.pyplot(fig_cv_bar)
        plt.close(fig_cv_bar)

    # ---------------------------------------------------------
    # TAB 3: DIAGNOSTICS & LEARNING CURVES
    # ---------------------------------------------------------
    with tab_learning:
        st.subheader("Model Learning Curves")
        st.info("Learning Curves help diagnose whether the models suffer from high bias (underfitting) or high variance (overfitting).")
        
        # Trigger button for generating learning curves
        gen_lc = st.button("📈 Generate / Refresh Learning Curves")
        
        if gen_lc or 'lc_fig' not in st.session_state:
            with st.spinner("Computing learning curves (fitting on subset training data)..."):
                fig_lc, axes_lc = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
                cv_lc = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                train_sizes_list = np.linspace(0.1, 1.0, 5)
                
                for i, (name, pipeline) in enumerate(fitted_pipelines.items()):
                    train_sizes_abs, train_scores, test_scores = learning_curve(
                        pipeline, X_train, y_train, cv=cv_lc, train_sizes=train_sizes_list,
                        scoring='accuracy', n_jobs=1, random_state=42
                    )
                    
                    train_mean = np.mean(train_scores, axis=1)
                    train_std = np.std(train_scores, axis=1)
                    test_mean = np.mean(test_scores, axis=1)
                    test_std = np.std(test_scores, axis=1)
                    
                    ax = axes_lc[i]
                    ax.plot(train_sizes_abs, train_mean, 'o-', color='#1a73e8', label='Training Score', linewidth=2)
                    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#1a73e8')
                    
                    ax.plot(train_sizes_abs, test_mean, 's-', color='#e67e22', label='Validation Score (CV)', linewidth=2)
                    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.15, color='#e67e22')
                    
                    ax.set_title(f'{name}', fontsize=14, fontweight='bold', pad=12)
                    ax.set_xlabel('Training Samples', fontsize=11)
                    if i == 0:
                        ax.set_ylabel('Accuracy Score', fontsize=11)
                    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
                    ax.set_ylim(0.70, 1.02)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    
                fig_lc.suptitle("Model Learning Curves (Bias vs. Variance Diagnosis)", fontsize=16, fontweight='bold', y=1.02)
                fig_lc.tight_layout()
                st.session_state['lc_fig'] = fig_lc
                
        st.pyplot(st.session_state['lc_fig'])

    # ---------------------------------------------------------
    # TAB 4: PERFORMANCE SHOWDOWN
    # ---------------------------------------------------------
    with tab_test:
        st.subheader("Held-Out Test Set Metrics")
        
        # Metric cards for Test Set F1-Scores
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"<div class='metric-container'><div class='metric-label'>Logistic Regression F1</div><div class='metric-value' style='color:#1a73e8;'>{test_results['Logistic Regression']['F1-Score']:.4f}</div></div>", unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"<div class='metric-container'><div class='metric-label'>Support Vector Machine F1</div><div class='metric-value' style='color:#9b59b6;'>{test_results['Support Vector Machine']['F1-Score']:.4f}</div></div>", unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"<div class='metric-container'><div class='metric-label'>Random Forest F1</div><div class='metric-value' style='color:#2ecc71;'>{test_results['Random Forest']['F1-Score']:.4f}</div></div>", unsafe_allow_html=True)
            
        col_fig1, col_fig2 = st.columns([3, 2])
        
        with col_fig1:
            st.write("**ROC Curve Comparison:**")
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6.5))
            colors = {
                'Logistic Regression': '#1a73e8',
                'Support Vector Machine': '#9b59b6',
                'Random Forest': '#2ecc71'
            }
            
            has_auc_curves = False
            for name, results in test_results.items():
                if results['FPR'] is not None and results['TPR'] is not None:
                    ax_roc.plot(results['FPR'], results['TPR'], color=colors[name], lw=3, label=f"{name} (AUC = {results['AUC']:.4f})")
                    has_auc_curves = True
            
            if has_auc_curves:
                ax_roc.plot([0, 1], [0, 1], color='#7f8c8d', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.5000)')
                ax_roc.set_xlim([-0.02, 1.02])
                ax_roc.set_ylim([-0.02, 1.02])
                ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='semibold')
                ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='semibold')
                ax_roc.set_title('ROC Curve Comparison (Test Set)', fontsize=13, fontweight='bold')
                ax_roc.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9)
                ax_roc.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig_roc)
            else:
                st.warning("ROC AUC calculations not supported for this dataset's test parameters.")
            plt.close(fig_roc)
            
        with col_fig2:
            st.write("**Side-by-Side Confusion Matrices:**")
            fig_cm, axes_cm = plt.subplots(3, 1, figsize=(6, 12))
            for idx, (name, results) in enumerate(test_results.items()):
                cm = results['ConfusionMatrix']
                ax = axes_cm[idx]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                            annot_kws={"size": 14, "weight": "bold"})
                ax.set_title(f'{name}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                
                # Check for standard target values
                try:
                    labels = [str(x) for x in np.unique(y_test)]
                    if len(labels) == 2:
                        ax.set_xticklabels([f"Class {labels[0]}", f"Class {labels[1]}"])
                        ax.set_yticklabels([f"Class {labels[0]}", f"Class {labels[1]}"], rotation=0)
                except Exception:
                    pass
            fig_cm.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

    # ---------------------------------------------------------
    # TAB 5: LIVE PREDICTION TOOL
    # ---------------------------------------------------------
    with tab_predict:
        st.subheader("🔮 Predict Target for Custom Features")
        st.write("Modify the inputs below to see predictions and probability scores generate in real-time from all three trained models.")
        
        # Let's generate user input controls for features.
        # To avoid showing 30 sliders for the Breast Cancer dataset, we can select the top 6 numeric features
        # and display inputs for them. For the remaining features, we will fill in the median.
        # If there are few features, we show inputs for all of them.
        
        max_inputs = 8
        important_numeric = numeric_cols[:max_inputs]
        left_over_numeric = numeric_cols[max_inputs:]
        
        st.markdown("##### Adjust Feature Values:")
        
        # Grid layout for inputs
        input_vals = {}
        cols_input = st.columns(3)
        
        # Render numeric inputs
        for idx, col_name in enumerate(important_numeric):
            col_ax = cols_input[idx % 3]
            min_val = float(df[col_name].min())
            max_val = float(df[col_name].max())
            mean_val = float(df[col_name].mean())
            
            # Draw standard sliders
            input_vals[col_name] = col_ax.slider(
                col_name,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val)/100.0
            )
            
        # Fill left-over features with training medians
        for col_name in left_over_numeric:
            input_vals[col_name] = float(df[col_name].median())
            
        # Draw categorical drop-downs
        for idx, col_name in enumerate(categorical_cols):
            col_ax = cols_input[(len(important_numeric) + idx) % 3]
            unique_cats = df[col_name].dropna().unique().tolist()
            input_vals[col_name] = col_ax.selectbox(
                col_name,
                unique_cats,
                index=0
            )
            
        # Convert dictionary to DataFrame (single row for prediction)
        input_df = pd.DataFrame([input_vals])
        
        st.write("---")
        st.markdown("##### Prediction Results:")
        
        cols_pred = st.columns(3)
        
        for idx, (name, pipeline) in enumerate(fitted_pipelines.items()):
            col_pred = cols_pred[idx]
            
            # Predict class and probability
            pred_class = pipeline.predict(input_df)[0]
            
            # Get probability if available
            pred_prob = None
            if hasattr(pipeline, "predict_proba"):
                pred_prob = pipeline.predict_proba(input_df)[0]
                
            with col_pred:
                st.markdown(f"<div style='border: 1px solid #eaeaea; border-radius: 8px; padding: 1rem; background-color:#fafafa; text-align:center;'>", unsafe_allow_html=True)
                st.markdown(f"**{name}**", unsafe_allow_html=True)
                
                # Check target styling
                # Standard breast cancer mapping: 0 = Malignant, 1 = Benign
                if target_col == 'target':
                    class_label = "Benign" if pred_class == 1 else "Malignant"
                    color_lbl = "#2ecc71" if pred_class == 1 else "#e74c3c"
                    st.markdown(f"<h3 style='color:{color_lbl}; margin:0;'>{class_label}</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color:#1a73e8; margin:0;'>Class {pred_class}</h3>", unsafe_allow_html=True)
                    
                if pred_prob is not None:
                    # Positive class is typically the second index (1) or class 1
                    try:
                        p_val = pred_prob[1]
                        st.markdown(f"Confidence: **{p_val:.2%}**", unsafe_allow_html=True)
                    except IndexError:
                        st.markdown(f"Confidence: **{pred_prob[0]:.2%}**", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

else:
    # Guide message if no dataset uploaded
    st.info("👈 Please upload a CSV classification dataset in the sidebar to run the custom Showdown!")
