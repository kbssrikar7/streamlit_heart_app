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

# Train-test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")

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
X_test = preprocessor.transform(X_test_raw)

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

xgb_preds = xgb_clf.predict(X_test)
xgb_probs = xgb_clf.predict_proba(X_test)[:, 1]
cb_preds = cb_clf.predict(X_test)
cb_probs = cb_clf.predict_proba(X_test)[:, 1]

xgb_metrics = {
    'accuracy': accuracy_score(y_test, xgb_preds),
    'precision': precision_score(y_test, xgb_preds),
    'recall': recall_score(y_test, xgb_preds),
    'f1': f1_score(y_test, xgb_preds),
    'roc_auc': roc_auc_score(y_test, xgb_probs)
}

cb_metrics = {
    'accuracy': accuracy_score(y_test, cb_preds),
    'precision': precision_score(y_test, cb_preds),
    'recall': recall_score(y_test, cb_preds),
    'f1': f1_score(y_test, cb_preds),
    'roc_auc': roc_auc_score(y_test, cb_probs)
}

print("\n=== Model Performance ===")
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

# Save ensemble weights (50-50 split)
ensemble_weights = {'w_xgb': 0.5, 'w_cat': 0.5}
with open(os.path.join(RESULTS_DIR, "ensemble_weights.json"), 'w') as f:
    json.dump(ensemble_weights, f)

print(f"\nAll artifacts saved to {RESULTS_DIR}/")
print("Models ready for Streamlit deployment!")

