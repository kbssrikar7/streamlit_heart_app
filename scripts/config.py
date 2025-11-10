"""
Configuration file for Heart Attack Risk Predictor
Centralizes all configuration values for easy maintenance
"""

# Model paths
MODEL_PATHS = {
    'xgb': 'models/xgb_model.joblib',
    'cat': 'models/cat_model.joblib',
    'preprocessor': 'models/preprocessor.joblib',
    'feature_info': 'models/feature_info.json',
    'weights': 'models/ensemble_weights.json'
}

# Risk classification thresholds
THRESHOLDS = {
    'standard': 0.5,
    'mapping_a': 0.3,
    'mapping_b': 0.7
}

# Input validation ranges
VALIDATION_RANGES = {
    'height': (120, 210),
    'weight': (40, 150),
    'bmi': (15, 50),
    'ap_hi': (90, 200),
    'ap_lo': (50, 120),
    'protein_level': (4.0, 12.0),
    'ejection_fraction': (30, 80),
    'age': (25, 90),
    'bp_diff': (20, 100)
}

# Normal/healthy ranges (for health summary)
NORMAL_RANGES = {
    'bmi': (18.5, 24.9),
    'bp_systolic': 120,
    'bp_diastolic': 80,
    'ejection_fraction': (55, 70),
    'protein_level': (6.0, 8.3),
    'pulse_pressure': (30, 60)
}

# Model performance metrics (for display)
MODEL_PERFORMANCE = {
    'accuracy': 0.85,
    'recall': 0.84,
    'roc_auc': 0.92,
    'precision': 0.86
}

# Confidence levels
CONFIDENCE_LEVELS = {
    'very_high': 80,
    'high': 60,
    'moderate': 40,
    'low': 0
}

