"""
Training script for Heart Attack Risk Prediction Model
Based on the notebook oneLastTime-2.ipynb
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configuration
DATA_PATH = "cardio_train_extended.csv"  # Update this path if needed
RESULTS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# Clean unrealistic values
df['ap_hi'] = df['ap_hi'].clip(lower=80, upper=250)
df['ap_lo'] = df['ap_lo'].clip(lower=40, upper=150)
df['BMI'] = df['BMI'].clip(lower=10, upper=60)
df['BP_diff'] = df['ap_hi'] - df['ap_lo']

# Select features and target
drop_cols = ['id', 'age', 'Reason']  # drop unused or text columns
X = df.drop(columns=drop_cols + ['cardio'])
y = df['cardio']

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts(normalize=True)}")

# Identify categorical vs numeric columns
cat_cols = [c for c in X.columns if X[c].dtype == 'object' or X[c].dtype.name == 'category']
num_cols = [c for c in X.columns if c not in cat_cols]

print(f"Numeric columns: {len(num_cols)}")
print(f"Categorical columns: {len(cat_cols)}")

# Train-test split (80-20)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Further split test set into validation (for weight optimization) and final test
# Paper: "optimized the weights on a validation set"
X_val_raw, X_test_final_raw, y_val, y_test_final = train_test_split(
    X_test_raw, y_test, test_size=0.5, random_state=RANDOM_STATE, stratify=y_test
)

print(f"Train: {X_train_raw.shape}, Validation: {X_val_raw.shape}, Test: {X_test_final_raw.shape}")

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
], remainder='drop')

# Fit and transform
print("Preprocessing data...")
X_train = preprocessor.fit_transform(X_train_raw)
X_val = preprocessor.transform(X_val_raw)
X_test = preprocessor.transform(X_test_final_raw)

# Get feature names after preprocessing
ohe_names = []
if cat_cols:
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    ohe_names = list(ohe.get_feature_names_out(cat_cols))
processed_feature_names = num_cols + ohe_names

print(f"Processed features: {len(processed_feature_names)}")

# Train XGBoost
print("\nTraining XGBoost...")
xgb_clf = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.08, max_depth=6,
    eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=1
)
xgb_clf.fit(X_train, y_train)

# Train CatBoost
print("Training CatBoost...")
cb_clf = CatBoostClassifier(
    iterations=600, depth=6, learning_rate=0.05, random_seed=RANDOM_STATE, verbose=0
)
cb_clf.fit(X_train, y_train)

# Evaluate models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# Get predictions on validation set for weight optimization
xgb_probs_val = xgb_clf.predict_proba(X_val)[:, 1]
cb_probs_val = cb_clf.predict_proba(X_val)[:, 1]

# Get predictions on test set for final evaluation
xgb_preds = xgb_clf.predict(X_test)
xgb_probs = xgb_clf.predict_proba(X_test)[:, 1]
cb_preds = cb_clf.predict(X_test)
cb_probs = cb_clf.predict_proba(X_test)[:, 1]

xgb_metrics = {
    'accuracy': accuracy_score(y_test_final, xgb_preds),
    'precision': precision_score(y_test_final, xgb_preds),
    'recall': recall_score(y_test_final, xgb_preds),
    'f1': f1_score(y_test_final, xgb_preds),
    'roc_auc': roc_auc_score(y_test_final, xgb_probs),
    'pr_auc': average_precision_score(y_test_final, xgb_probs)
}

cb_metrics = {
    'accuracy': accuracy_score(y_test_final, cb_preds),
    'precision': precision_score(y_test_final, cb_preds),
    'recall': recall_score(y_test_final, cb_preds),
    'f1': f1_score(y_test_final, cb_preds),
    'roc_auc': roc_auc_score(y_test_final, cb_probs),
    'pr_auc': average_precision_score(y_test_final, cb_probs)
}

print("\n=== Model Performance (Test Set) ===")
print("\nXGBoost Metrics:")
for k, v in xgb_metrics.items():
    print(f"  {k}: {v:.4f}")

print("\nCatBoost Metrics:")
for k, v in cb_metrics.items():
    print(f"  {k}: {v:.4f}")

# Save models and artifacts
print("\nSaving models and artifacts...")
joblib.dump(preprocessor, os.path.join(RESULTS_DIR, "preprocessor.joblib"))
joblib.dump(xgb_clf, os.path.join(RESULTS_DIR, "xgb_model.joblib"))
joblib.dump(cb_clf, os.path.join(RESULTS_DIR, "cat_model.joblib"))

# Save feature information
import json
feature_info = {
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'processed_feature_names': processed_feature_names,
    'drop_cols': drop_cols
}
with open(os.path.join(RESULTS_DIR, "feature_info.json"), 'w') as f:
    json.dump(feature_info, f, indent=2)

# Ensemble Weight Optimization (Paper Section III.F)
# "We optimized the weights on a validation set (grid search over 0.0–1.0 in steps of 0.05) 
# by maximizing ROC-AUC and F1 on the hold-out fold"
print("\n=== Optimizing Ensemble Weights ===")
print("Grid search over weights (0.0 to 1.0, steps of 0.05)...")

best_roc_auc = 0
best_f1 = 0
best_weights_roc = (0.5, 0.5)
best_weights_f1 = (0.5, 0.5)
weight_results = []

# Grid search over weights
weights = np.arange(0.0, 1.01, 0.05)
for w_xgb in weights:
    w_cat = 1.0 - w_xgb
    # Ensemble probability on validation set
    ensemble_probs_val = w_xgb * xgb_probs_val + w_cat * cb_probs_val
    ensemble_preds_val = (ensemble_probs_val >= 0.5).astype(int)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_val, ensemble_probs_val)
    f1 = f1_score(y_val, ensemble_preds_val)
    
    weight_results.append({
        'w_xgb': w_xgb,
        'w_cat': w_cat,
        'roc_auc': roc_auc,
        'f1': f1
    })
    
    # Track best weights for ROC-AUC
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_weights_roc = (w_xgb, w_cat)
    
    # Track best weights for F1
    if f1 > best_f1:
        best_f1 = f1
        best_weights_f1 = (w_xgb, w_cat)

# Paper says "maximizing ROC-AUC and F1" - use average or prioritize ROC-AUC
# Using ROC-AUC as primary (as it's more common for ensemble optimization)
best_weights = best_weights_roc
print(f"\nBest weights (ROC-AUC): w_xgb={best_weights[0]:.2f}, w_cat={best_weights[1]:.2f}, ROC-AUC={best_roc_auc:.4f}")
print(f"Best weights (F1): w_xgb={best_weights_f1[0]:.2f}, w_cat={best_weights_f1[1]:.2f}, F1={best_f1:.4f}")
print(f"Selected weights: w_xgb={best_weights[0]:.2f}, w_cat={best_weights[1]:.2f}")

# Save optimized weights
ensemble_weights = {'w_xgb': float(best_weights[0]), 'w_cat': float(best_weights[1])}
with open(os.path.join(RESULTS_DIR, "ensemble_weights.json"), 'w') as f:
    json.dump(ensemble_weights, f, indent=2)

# Save weight optimization results
weight_df = pd.DataFrame(weight_results)
weight_df.to_csv(os.path.join(RESULTS_DIR, "weight_optimization_results.csv"), index=False)
print(f"Weight optimization results saved to {RESULTS_DIR}/weight_optimization_results.csv")

# Calculate ensemble metrics on test set with optimized weights
ensemble_probs = ensemble_weights['w_xgb'] * xgb_probs + ensemble_weights['w_cat'] * cb_probs
ensemble_preds = (ensemble_probs >= 0.5).astype(int)

ensemble_metrics = {
    'accuracy': accuracy_score(y_test_final, ensemble_preds),
    'precision': precision_score(y_test_final, ensemble_preds),
    'recall': recall_score(y_test_final, ensemble_preds),
    'f1': f1_score(y_test_final, ensemble_preds),
    'roc_auc': roc_auc_score(y_test_final, ensemble_probs),
    'pr_auc': average_precision_score(y_test_final, ensemble_probs)
}

print(f"\n=== Ensemble Performance (Test Set) ===")
print(f"Using optimized weights: w_xgb={ensemble_weights['w_xgb']:.2f}, w_cat={ensemble_weights['w_cat']:.2f}")
for k, v in ensemble_metrics.items():
    print(f"  {k}: {v:.4f}")

print(f"\nAll artifacts saved to {RESULTS_DIR}/")
print("Models ready for Streamlit deployment!")

