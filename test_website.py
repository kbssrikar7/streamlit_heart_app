"""
Website Functionality Test Suite
Tests website accessibility, model files, and code logic
"""

import sys
import os
import json
from pathlib import Path
import subprocess

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
print("WEBSITE FUNCTIONALITY TEST SUITE")
print("=" * 70)
print()

# ============================================================================
# TEST 1: File Structure
# ============================================================================
print("TEST CATEGORY 1: File Structure")
print("-" * 70)

test("app.py exists", Path("app.py").exists())
test("requirements.txt exists", Path("requirements.txt").exists())
test("Dockerfile exists", Path("Dockerfile").exists())
test("docker-compose.yml exists", Path("docker-compose.yml").exists())
test("Dataset file exists", Path("cardio_train_extended.csv").exists())
test("Models directory exists", Path("models").exists())
print()

# ============================================================================
# TEST 2: Model Files
# ============================================================================
print("TEST CATEGORY 2: Model Files")
print("-" * 70)

models_dir = Path("models")
test("XGBoost model file exists", (models_dir / "xgb_model.joblib").exists())
test("CatBoost model file exists", (models_dir / "cat_model.joblib").exists())
test("Preprocessor file exists", (models_dir / "preprocessor.joblib").exists())
test("Feature info file exists", (models_dir / "feature_info.json").exists())
test("Ensemble weights file exists", (models_dir / "ensemble_weights.json").exists())

# Check file sizes (should not be empty)
if (models_dir / "xgb_model.joblib").exists():
    xgb_size = (models_dir / "xgb_model.joblib").stat().st_size
    test("XGBoost model file is not empty", xgb_size > 1000, f"Size: {xgb_size} bytes")

if (models_dir / "cat_model.joblib").exists():
    cat_size = (models_dir / "cat_model.joblib").stat().st_size
    test("CatBoost model file is not empty", cat_size > 1000, f"Size: {cat_size} bytes")
print()

# ============================================================================
# TEST 3: Configuration Files
# ============================================================================
print("TEST CATEGORY 3: Configuration Files")
print("-" * 70)

# Check ensemble weights
if (models_dir / "ensemble_weights.json").exists():
    try:
        with open(models_dir / "ensemble_weights.json", 'r') as f:
            weights = json.load(f)
        w_xgb = weights.get('w_xgb', 0)
        w_cat = weights.get('w_cat', 0)
        test("Ensemble weights are valid JSON", True)
        test("Ensemble weights sum to 1", abs(w_xgb + w_cat - 1.0) < 0.01)
        test("Ensemble weights are 50/50 (paper spec)", 
             abs(w_xgb - 0.5) < 0.01 and abs(w_cat - 0.5) < 0.01,
             f"Found: {w_xgb:.2f}/{w_cat:.2f}")
    except Exception as e:
        test("Ensemble weights are valid JSON", False, str(e))

# Check feature info
if (models_dir / "feature_info.json").exists():
    try:
        with open(models_dir / "feature_info.json", 'r') as f:
            feature_info = json.load(f)
        test("Feature info is valid JSON", True)
        test("Feature info has num_cols", 'num_cols' in feature_info)
        test("Feature info has cat_cols", 'cat_cols' in feature_info)
        test("Protein_Level in num_cols", 'Protein_Level' in feature_info.get('num_cols', []))
        test("Ejection_Fraction in num_cols", 'Ejection_Fraction' in feature_info.get('num_cols', []))
    except Exception as e:
        test("Feature info is valid JSON", False, str(e))
print()

# ============================================================================
# TEST 4: Code Syntax Check
# ============================================================================
print("TEST CATEGORY 4: Code Syntax Check")
print("-" * 70)

# Check if app.py can be parsed
try:
    with open("app.py", 'r') as f:
        code = f.read()
    compile(code, "app.py", "exec")
    test("app.py has valid Python syntax", True)
except SyntaxError as e:
    test("app.py has valid Python syntax", False, f"Syntax error: {str(e)}")

# Check for critical imports
required_imports = ['streamlit', 'pandas', 'numpy', 'joblib', 'matplotlib', 'shap']
for imp in required_imports:
    if imp in code:
        test(f"app.py imports {imp}", True)
    else:
        test(f"app.py imports {imp}", False, f"{imp} not found in imports")
print()

# ============================================================================
# TEST 5: Code Logic Checks
# ============================================================================
print("TEST CATEGORY 5: Code Logic Checks")
print("-" * 70)

# Check for correct Smoker_Alcoholic formula
if 'smoker_alcoholic' in code and 'smoke == 1 and alco == 1' in code:
    test("Smoker_Alcoholic uses AND (not OR)", True)
elif 'smoke == 1 or alco == 1' in code:
    test("Smoker_Alcoholic uses AND (not OR)", False, "Found OR instead of AND")
else:
    test("Smoker_Alcoholic formula exists", 'smoker_alcoholic' in code)

# Check for MAP formula
if 'map_value' in code and ('ap_lo + (bp_diff / 3)' in code or '(ap_hi + 2' in code):
    test("MAP formula exists in code", True)
else:
    test("MAP formula exists in code", False)

# Check for Lifestyle Score formula
if 'lifestyle_score' in code and 'active - (smoke + alco)' in code:
    test("Lifestyle Score formula exists", True)
else:
    test("Lifestyle Score formula exists", False)

# Check for Risk Age formula
if 'risk_age' in code and 'bmi / 5' in code:
    test("Risk Age formula exists", True)
else:
    test("Risk Age formula exists", False)

# Check for BP swap logic
if 'ap_lo > ap_hi' in code:
    test("BP swap logic exists", True)
else:
    test("BP swap logic exists", False)

# Check for input validation
if 'validation_warnings' in code:
    test("Input validation exists", True)
else:
    test("Input validation exists", False)
print()

# ============================================================================
# TEST 6: Dataset Verification
# ============================================================================
print("TEST CATEGORY 6: Dataset Verification")
print("-" * 70)

if Path("cardio_train_extended.csv").exists():
    try:
        # Check file size
        file_size = Path("cardio_train_extended.csv").stat().st_size
        test("Dataset file is not empty", file_size > 1000000, f"Size: {file_size} bytes")
        
        # Check first few lines to verify structure
        with open("cardio_train_extended.csv", 'r') as f:
            first_line = f.readline()
            headers = first_line.strip().split(',')
        
        required_columns = ['Protein_Level', 'Ejection_Fraction', 'Smoker_Alcoholic', 
                           'cardio', 'age_years', 'BMI', 'ap_hi', 'ap_lo']
        for col in required_columns:
            test(f"Dataset has {col} column", col in headers)
        
        # Check dataset has reasonable number of columns
        test("Dataset has expected number of columns", len(headers) >= 30, 
             f"Found {len(headers)} columns")
    except Exception as e:
        test("Dataset is readable", False, str(e))
print()

# ============================================================================
# TEST 7: Website Accessibility
# ============================================================================
print("TEST CATEGORY 7: Website Accessibility")
print("-" * 70)

# Check if Docker container is running
try:
    result = subprocess.run(['docker', 'ps', '--filter', 'name=heart-attack-predictor', 
                            '--format', '{{.Status}}'], 
                           capture_output=True, text=True, timeout=5)
    if 'Up' in result.stdout:
        test("Docker container is running", True)
    else:
        test("Docker container is running", False, "Container not running")
except Exception as e:
    test("Docker container check", False, str(e))

# Check if port 8501 is accessible
try:
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('localhost', 8501))
    sock.close()
    test("Port 8501 is accessible", result == 0, "Port 8501 not accessible")
except Exception as e:
    test("Port 8501 accessibility check", False, str(e))
print()

# ============================================================================
# TEST 8: Requirements File
# ============================================================================
print("TEST CATEGORY 8: Requirements File")
print("-" * 70)

if Path("requirements.txt").exists():
    try:
        with open("requirements.txt", 'r') as f:
            requirements = f.read()
        
        required_packages = ['streamlit', 'pandas', 'numpy', 'scikit-learn', 
                           'xgboost', 'catboost', 'shap', 'matplotlib']
        for pkg in required_packages:
            if pkg in requirements.lower():
                test(f"requirements.txt includes {pkg}", True)
            else:
                test(f"requirements.txt includes {pkg}", False)
    except Exception as e:
        test("requirements.txt is readable", False, str(e))
print()

# ============================================================================
# TEST 9: Documentation
# ============================================================================
print("TEST CATEGORY 9: Documentation")
print("-" * 70)

test("README.md exists", Path("README.md").exists())
test("Dockerfile exists", Path("Dockerfile").exists())
test("docker-compose.yml exists", Path("docker-compose.yml").exists())
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

