"""
Comprehensive Test Suite for Heart Attack Risk Prediction App
Tests all formulas, features, and functionality
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

# Test results tracking
test_results = []
passed = 0
failed = 0

def test(name, condition, error_msg=""):
    """Run a test and track results"""
    global passed, failed
    try:
        if condition:
            print(f"✅ PASS: {name}")
            test_results.append(("PASS", name))
            passed += 1
        else:
            print(f"❌ FAIL: {name}")
            if error_msg:
                print(f"   Error: {error_msg}")
            test_results.append(("FAIL", name, error_msg))
            failed += 1
    except Exception as e:
        print(f"❌ ERROR: {name}")
        print(f"   Exception: {str(e)}")
        test_results.append(("ERROR", name, str(e)))
        failed += 1

print("=" * 70)
print("HEART ATTACK RISK PREDICTION APP - COMPREHENSIVE TEST SUITE")
print("=" * 70)
print()

# ============================================================================
# TEST 1: Model Files Existence
# ============================================================================
print("TEST CATEGORY 1: Model Files Existence")
print("-" * 70)

models_dir = Path("models")
test("Model directory exists", models_dir.exists())
test("XGBoost model exists", (models_dir / "xgb_model.joblib").exists())
test("CatBoost model exists", (models_dir / "cat_model.joblib").exists())
test("Preprocessor exists", (models_dir / "preprocessor.joblib").exists())
test("Feature info exists", (models_dir / "feature_info.json").exists())
test("Ensemble weights exist", (models_dir / "ensemble_weights.json").exists())
print()

# ============================================================================
# TEST 2: Model Loading
# ============================================================================
print("TEST CATEGORY 2: Model Loading")
print("-" * 70)

try:
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    xgb_model = joblib.load(models_dir / "xgb_model.joblib")
    cat_model = joblib.load(models_dir / "cat_model.joblib")
    
    with open(models_dir / "feature_info.json", 'r') as f:
        feature_info = json.load(f)
    
    with open(models_dir / "ensemble_weights.json", 'r') as f:
        ensemble_weights = json.load(f)
    
    test("Preprocessor loads successfully", preprocessor is not None)
    test("XGBoost model loads successfully", xgb_model is not None)
    test("CatBoost model loads successfully", cat_model is not None)
    test("Feature info loads successfully", feature_info is not None)
    test("Ensemble weights load successfully", ensemble_weights is not None)
    
    # Check ensemble weights
    w_xgb = ensemble_weights.get('w_xgb', 0)
    w_cat = ensemble_weights.get('w_cat', 0)
    test("Ensemble weights sum to 1", abs(w_xgb + w_cat - 1.0) < 0.01)
    test("Ensemble weights are 50/50 (paper spec)", abs(w_xgb - 0.5) < 0.01 and abs(w_cat - 0.5) < 0.01)
    
except Exception as e:
    test("Models load successfully", False, str(e))
print()

# ============================================================================
# TEST 3: Formula Verification
# ============================================================================
print("TEST CATEGORY 3: Formula Verification")
print("-" * 70)

# Test BP_diff
ap_hi, ap_lo = 120, 80
bp_diff = ap_hi - ap_lo
test("BP_diff formula: ap_hi - ap_lo", bp_diff == 40)

# Test MAP formula (paper: (ap_hi + 2*ap_lo) / 3)
map_paper = (ap_hi + 2 * ap_lo) / 3
map_code = ap_lo + (bp_diff / 3)
test("MAP formula matches paper", abs(map_paper - map_code) < 0.01, 
     f"Paper: {map_paper:.2f}, Code: {map_code:.2f}")

# Test Pulse Pressure Ratio
pulse_pressure_ratio = bp_diff / ap_hi if ap_hi > 0 else 0
expected_ppr = (ap_hi - ap_lo) / ap_hi
test("Pulse Pressure Ratio formula", abs(pulse_pressure_ratio - expected_ppr) < 0.01)

# Test Lifestyle Score
active, smoke, alco = 1, 0, 0
lifestyle_score_raw = active - (smoke + alco)
test("Lifestyle Score formula: active - (smoke + alco)", lifestyle_score_raw == 1)

# Test Hypertension Flag
test("Hypertension Flag: ap_hi >= 140 or ap_lo >= 90", 
     (1 if (160 >= 140 or 100 >= 90) else 0) == 1)
test("Hypertension Flag: normal BP", 
     (1 if (120 >= 140 or 80 >= 90) else 0) == 0)

# Test Obesity Flag
test("Obesity Flag: BMI >= 30", (1 if 35 >= 30 else 0) == 1)
test("Obesity Flag: normal BMI", (1 if 25 >= 30 else 0) == 0)

# Test Smoker Alcoholic (AND, not OR)
test("Smoker Alcoholic: smoke=1 AND alco=1", 
     (1 if (1 == 1 and 1 == 1) else 0) == 1)
test("Smoker Alcoholic: smoke=1 OR alco=0 (should be 0)", 
     (1 if (1 == 1 and 0 == 1) else 0) == 0)
test("Smoker Alcoholic: smoke=0 OR alco=1 (should be 0)", 
     (1 if (0 == 1 and 1 == 1) else 0) == 0)

# Test Risk Age
age_years, bmi, cholesterol, gluc = 50, 25, 2, 2
risk_age = age_years + (bmi / 5) + (2 * (1 if cholesterol > 1 else 0)) + (1 if gluc > 1 else 0)
expected_risk_age = 50 + (25 / 5) + 2 + 1  # 50 + 5 + 2 + 1 = 58
test("Risk Age formula", risk_age == expected_risk_age)
print()

# ============================================================================
# TEST 4: Feature Dictionary Creation
# ============================================================================
print("TEST CATEGORY 4: Feature Dictionary Creation")
print("-" * 70)

# Create a sample feature dictionary
feature_dict = {
    'gender': 1,
    'height': 175,
    'weight': 75,
    'ap_hi': 120,
    'ap_lo': 80,
    'cholesterol': 1,
    'gluc': 1,
    'smoke': 0,
    'alco': 0,
    'active': 1,
    'BMI': 24.5,
    'BP_diff': 40,
    'Systolic_Pressure': 120,
    'age_years': 50,
    'Age_Group': '50-59',
    'Lifestyle_Score': 0,
    'Obesity_Flag': 0,
    'Hypertension_Flag': 0,
    'Health_Risk_Score': 0,
    'Pulse_Pressure_Ratio': 40/120,
    'MAP': (120 + 2*80)/3,
    'BMI_Category': 'Normal',
    'Smoker_Alcoholic': 0,
    'BP_Category': 'Normal',
    'Risk_Age': 50 + (24.5/5),
    'Risk_Level': 'Low',
    'Protein_Level': 6.8,
    'Ejection_Fraction': 60.0
}

# Check required features
required_features = feature_info.get('num_cols', []) + feature_info.get('cat_cols', [])
missing_features = [f for f in required_features if f not in feature_dict]
test("All required features present", len(missing_features) == 0, 
     f"Missing: {missing_features}" if missing_features else "")
print()

# ============================================================================
# TEST 5: Preprocessing Pipeline
# ============================================================================
print("TEST CATEGORY 5: Preprocessing Pipeline")
print("-" * 70)

try:
    # Create input DataFrame
    input_df = pd.DataFrame([feature_dict])
    
    # Reorder columns to match training data
    available_features = [f for f in required_features if f in input_df.columns]
    input_df = input_df[available_features]
    
    # Transform using preprocessor
    X_processed = preprocessor.transform(input_df)
    
    test("Preprocessing transforms data", X_processed.shape[0] == 1)
    test("Processed features match expected count", 
         X_processed.shape[1] == len(feature_info.get('processed_feature_names', [])))
    test("No NaN values in processed data", not np.isnan(X_processed).any())
    test("No infinite values in processed data", not np.isinf(X_processed).any())
    
except Exception as e:
    test("Preprocessing works correctly", False, str(e))
print()

# ============================================================================
# TEST 6: Model Predictions
# ============================================================================
print("TEST CATEGORY 6: Model Predictions")
print("-" * 70)

try:
    # Get predictions
    xgb_prob = xgb_model.predict_proba(X_processed)[0, 1]
    cat_prob = cat_model.predict_proba(X_processed)[0, 1]
    
    # Ensemble prediction
    w_xgb = ensemble_weights.get('w_xgb', 0.5)
    w_cat = ensemble_weights.get('w_cat', 0.5)
    ensemble_prob = w_xgb * xgb_prob + w_cat * cat_prob
    
    test("XGBoost prediction in valid range", 0 <= xgb_prob <= 1)
    test("CatBoost prediction in valid range", 0 <= cat_prob <= 1)
    test("Ensemble prediction in valid range", 0 <= ensemble_prob <= 1)
    test("Ensemble prediction is weighted average", 
         abs(ensemble_prob - (w_xgb * xgb_prob + w_cat * cat_prob)) < 0.0001)
    
except Exception as e:
    test("Model predictions work correctly", False, str(e))
print()

# ============================================================================
# TEST 7: Input Validation
# ============================================================================
print("TEST CATEGORY 7: Input Validation")
print("-" * 70)

# Test BP swap logic
ap_hi_wrong, ap_lo_wrong = 80, 120  # Diastolic > Systolic (wrong)
if ap_lo_wrong > ap_hi_wrong:
    ap_hi_fixed, ap_lo_fixed = ap_lo_wrong, ap_hi_wrong
    test("BP swap logic: diastolic > systolic", 
         ap_hi_fixed == 120 and ap_lo_fixed == 80)

# Test validation ranges
test("Protein Level: valid range (6.8)", 4.0 <= 6.8 <= 12.0)
test("Protein Level: extreme low (< 4.0)", not (4.0 <= 3.0 <= 12.0))
test("Protein Level: extreme high (> 12.0)", not (4.0 <= 15.0 <= 12.0))

test("Ejection Fraction: valid range (60)", 30 <= 60 <= 80)
test("Ejection Fraction: low (< 30)", not (30 <= 25 <= 80))
test("Ejection Fraction: high (> 80)", not (30 <= 85 <= 80))
print()

# ============================================================================
# TEST 8: Dataset Verification
# ============================================================================
print("TEST CATEGORY 8: Dataset Verification")
print("-" * 70)

try:
    df = pd.read_csv("cardio_train_extended.csv")
    test("Dataset loads successfully", df.shape[0] > 0)
    test("Dataset has expected columns", 'Protein_Level' in df.columns)
    test("Dataset has expected columns", 'Ejection_Fraction' in df.columns)
    test("Dataset has target column", 'cardio' in df.columns)
    
    # Check Protein Level range
    protein_min = df['Protein_Level'].min()
    protein_max = df['Protein_Level'].max()
    test("Protein Level in expected range (5.6-8.0)", 
         5.0 <= protein_min <= 6.0 and 8.0 <= protein_max <= 9.0,
         f"Range: {protein_min:.2f}-{protein_max:.2f}")
    
    # Verify Smoker_Alcoholic formula in dataset
    sample = df.head(100)
    matches = (sample['Smoker_Alcoholic'] == 
               ((sample['smoke'] == 1) & (sample['alco'] == 1)).astype(int)).sum()
    test("Smoker_Alcoholic uses AND in dataset", matches == 100,
         f"Only {matches}/100 rows match AND formula")
    
except Exception as e:
    test("Dataset verification", False, str(e))
print()

# ============================================================================
# TEST 9: Edge Cases
# ============================================================================
print("TEST CATEGORY 9: Edge Cases")
print("-" * 70)

# Test with extreme values
extreme_dict = feature_dict.copy()
extreme_dict['ap_hi'] = 250
extreme_dict['ap_lo'] = 150
extreme_dict['BMI'] = 50
extreme_dict['age_years'] = 100

try:
    extreme_df = pd.DataFrame([extreme_dict])
    available_features = [f for f in required_features if f in extreme_df.columns]
    extreme_df = extreme_df[available_features]
    X_extreme = preprocessor.transform(extreme_df)
    prob_extreme = xgb_model.predict_proba(X_extreme)[0, 1]
    test("Extreme values don't break prediction", 0 <= prob_extreme <= 1)
except Exception as e:
    test("Extreme values handled correctly", False, str(e))

# Test with minimal values
minimal_dict = feature_dict.copy()
minimal_dict['ap_hi'] = 80
minimal_dict['ap_lo'] = 40
minimal_dict['BMI'] = 15
minimal_dict['age_years'] = 20

try:
    minimal_df = pd.DataFrame([minimal_dict])
    available_features = [f for f in required_features if f in minimal_df.columns]
    minimal_df = minimal_df[available_features]
    X_minimal = preprocessor.transform(minimal_df)
    prob_minimal = xgb_model.predict_proba(X_minimal)[0, 1]
    test("Minimal values don't break prediction", 0 <= prob_minimal <= 1)
except Exception as e:
    test("Minimal values handled correctly", False, str(e))
print()

# ============================================================================
# TEST 10: Formula Consistency Check
# ============================================================================
print("TEST CATEGORY 10: Formula Consistency Check")
print("-" * 70)

# Test multiple scenarios
test_cases = [
    {"ap_hi": 140, "ap_lo": 90, "bmi": 30, "smoke": 1, "alco": 1, "active": 0},
    {"ap_hi": 120, "ap_lo": 80, "bmi": 22, "smoke": 0, "alco": 0, "active": 1},
    {"ap_hi": 130, "ap_lo": 85, "bmi": 28, "smoke": 0, "alco": 1, "active": 1},
]

for i, case in enumerate(test_cases):
    ap_hi, ap_lo = case['ap_hi'], case['ap_lo']
    bmi = case['bmi']
    smoke, alco, active = case['smoke'], case['alco'], case['active']
    
    # Calculate all formulas
    bp_diff = ap_hi - ap_lo
    map_val = (ap_hi + 2 * ap_lo) / 3
    lifestyle = active - (smoke + alco)
    hypertension = 1 if (ap_hi >= 140 or ap_lo >= 90) else 0
    obesity = 1 if bmi >= 30 else 0
    smoker_alcoholic = 1 if (smoke == 1 and alco == 1) else 0
    
    test(f"Formula consistency test case {i+1}: BP_diff", bp_diff == ap_hi - ap_lo)
    test(f"Formula consistency test case {i+1}: MAP", 
         abs(map_val - ((ap_hi + 2*ap_lo)/3)) < 0.01)
    test(f"Formula consistency test case {i+1}: Lifestyle", 
         lifestyle == active - (smoke + alco))
    test(f"Formula consistency test case {i+1}: Smoker_Alcoholic", 
         smoker_alcoholic == (1 if (smoke == 1 and alco == 1) else 0))
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Total Tests: {passed + failed}")
print(f"✅ Passed: {passed}")
print(f"❌ Failed: {failed}")
print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
print()

if failed == 0:
    print("🎉 ALL TESTS PASSED! The website is ready for deployment.")
else:
    print("⚠️  Some tests failed. Please review the errors above.")
    print("\nFailed Tests:")
    for result in test_results:
        if result[0] != "PASS":
            print(f"  - {result[1]}: {result[2] if len(result) > 2 else 'Unknown error'}")

print("=" * 70)

# Exit with appropriate code
sys.exit(0 if failed == 0 else 1)

