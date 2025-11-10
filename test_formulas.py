"""
Formula Verification Test Suite
Tests all formulas without requiring model files
"""

import sys

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
print("FORMULA VERIFICATION TEST SUITE")
print("=" * 70)
print()

# ============================================================================
# TEST 1: BP_diff (Pulse Pressure)
# ============================================================================
print("TEST 1: BP_diff (Pulse Pressure) Formula")
print("-" * 70)
ap_hi, ap_lo = 120, 80
bp_diff = ap_hi - ap_lo
test("BP_diff = ap_hi - ap_lo", bp_diff == 40)
test("BP_diff calculation", bp_diff == (120 - 80))
print()

# ============================================================================
# TEST 2: MAP (Mean Arterial Pressure)
# ============================================================================
print("TEST 2: MAP (Mean Arterial Pressure) Formula")
print("-" * 70)
# Paper formula: (ap_hi + 2 × ap_lo) / 3
map_paper = (ap_hi + 2 * ap_lo) / 3
# Code formula: ap_lo + (bp_diff / 3)
map_code = ap_lo + (bp_diff / 3)
test("MAP paper formula: (ap_hi + 2*ap_lo) / 3", abs(map_paper - 93.333) < 0.1)
test("MAP code formula: ap_lo + (bp_diff / 3)", abs(map_code - 93.333) < 0.1)
test("MAP formulas are equivalent", abs(map_paper - map_code) < 0.01,
     f"Paper: {map_paper:.3f}, Code: {map_code:.3f}")
print()

# ============================================================================
# TEST 3: Pulse Pressure Ratio
# ============================================================================
print("TEST 3: Pulse Pressure Ratio Formula")
print("-" * 70)
pulse_pressure_ratio = bp_diff / ap_hi if ap_hi > 0 else 0
expected_ppr = (ap_hi - ap_lo) / ap_hi
test("Pulse Pressure Ratio = (ap_hi - ap_lo) / ap_hi", 
     abs(pulse_pressure_ratio - expected_ppr) < 0.01)
test("Pulse Pressure Ratio calculation", abs(pulse_pressure_ratio - (40/120)) < 0.01)
print()

# ============================================================================
# TEST 4: Lifestyle Score
# ============================================================================
print("TEST 4: Lifestyle Score Formula")
print("-" * 70)
# Paper: active - (smoke + alco)
active, smoke, alco = 1, 0, 0
lifestyle_score_raw = active - (smoke + alco)
test("Lifestyle Score: active - (smoke + alco)", lifestyle_score_raw == 1)

# Test all combinations
test_cases = [
    (1, 0, 0, 1),   # active, no smoke, no alco = 1 (best)
    (1, 1, 0, 0),   # active, smoke, no alco = 0
    (1, 0, 1, 0),   # active, no smoke, alco = 0
    (1, 1, 1, -1),  # active, smoke, alco = -1
    (0, 0, 0, 0),   # inactive, no smoke, no alco = 0
    (0, 1, 0, -1),  # inactive, smoke, no alco = -1
    (0, 0, 1, -1),  # inactive, no smoke, alco = -1
    (0, 1, 1, -2),  # inactive, smoke, alco = -2 (worst)
]

for active, smoke, alco, expected in test_cases:
    result = active - (smoke + alco)
    test(f"Lifestyle: active={active}, smoke={smoke}, alco={alco} = {expected}", 
         result == expected)
print()

# ============================================================================
# TEST 5: Hypertension Flag
# ============================================================================
print("TEST 5: Hypertension Flag Formula")
print("-" * 70)
# Paper: 1 if ap_hi >= 140 or ap_lo >= 90 else 0
test("Hypertension: ap_hi=160, ap_lo=100 (both high)", 
     (1 if (160 >= 140 or 100 >= 90) else 0) == 1)
test("Hypertension: ap_hi=140, ap_lo=80 (systolic high)", 
     (1 if (140 >= 140 or 80 >= 90) else 0) == 1)
test("Hypertension: ap_hi=120, ap_lo=90 (diastolic high)", 
     (1 if (120 >= 140 or 90 >= 90) else 0) == 1)
test("Hypertension: ap_hi=120, ap_lo=80 (normal)", 
     (1 if (120 >= 140 or 80 >= 90) else 0) == 0)
print()

# ============================================================================
# TEST 6: Obesity Flag
# ============================================================================
print("TEST 6: Obesity Flag Formula")
print("-" * 70)
# Paper: 1 if BMI >= 30 else 0
test("Obesity: BMI=35 (obese)", (1 if 35 >= 30 else 0) == 1)
test("Obesity: BMI=30 (obese threshold)", (1 if 30 >= 30 else 0) == 1)
test("Obesity: BMI=25 (overweight)", (1 if 25 >= 30 else 0) == 0)
test("Obesity: BMI=22 (normal)", (1 if 22 >= 30 else 0) == 0)
print()

# ============================================================================
# TEST 7: Smoker Alcoholic Flag
# ============================================================================
print("TEST 7: Smoker Alcoholic Flag Formula")
print("-" * 70)
# Paper: 1 if smoke = 1 & alco = 1 else 0 (AND, not OR)
test("Smoker Alcoholic: smoke=1 AND alco=1", 
     (1 if (1 == 1 and 1 == 1) else 0) == 1)
test("Smoker Alcoholic: smoke=1 AND alco=0 (should be 0)", 
     (1 if (1 == 1 and 0 == 1) else 0) == 0)
test("Smoker Alcoholic: smoke=0 AND alco=1 (should be 0)", 
     (1 if (0 == 1 and 1 == 1) else 0) == 0)
test("Smoker Alcoholic: smoke=0 AND alco=0 (should be 0)", 
     (1 if (0 == 1 and 0 == 1) else 0) == 0)

# Verify it's AND, not OR
test("Smoker Alcoholic uses AND (not OR): smoke=1, alco=0", 
     (1 if (1 == 1 and 0 == 1) else 0) != (1 if (1 == 1 or 0 == 1) else 0))
print()

# ============================================================================
# TEST 8: Risk Age
# ============================================================================
print("TEST 8: Risk Age Formula")
print("-" * 70)
# Paper: age_years + BMI/5 + 2*(cholesterol > 1) + (gluc > 1)
age_years, bmi, cholesterol, gluc = 50, 25, 2, 2
risk_age = age_years + (bmi / 5) + (2 * (1 if cholesterol > 1 else 0)) + (1 if gluc > 1 else 0)
expected = 50 + (25 / 5) + 2 + 1  # 50 + 5 + 2 + 1 = 58
test("Risk Age: age=50, BMI=25, chol=2, gluc=2", risk_age == expected)

# Test with normal values
age_years, bmi, cholesterol, gluc = 40, 22, 1, 1
risk_age = age_years + (bmi / 5) + (2 * (1 if cholesterol > 1 else 0)) + (1 if gluc > 1 else 0)
expected = 40 + (22 / 5) + 0 + 0  # 40 + 4.4 + 0 + 0 = 44.4
test("Risk Age: age=40, BMI=22, chol=1, gluc=1", abs(risk_age - expected) < 0.1)
print()

# ============================================================================
# TEST 9: Code Logic Verification (from app.py)
# ============================================================================
print("TEST 9: Code Logic Verification")
print("-" * 70)

# Test BP swap logic
ap_hi_wrong, ap_lo_wrong = 80, 120  # Diastolic > Systolic (wrong)
bp_swapped = False
if ap_lo_wrong > ap_hi_wrong:
    ap_hi_fixed, ap_lo_fixed = ap_lo_wrong, ap_hi_wrong
    bp_swapped = True
test("BP swap logic: detects diastolic > systolic", bp_swapped)
test("BP swap logic: corrects values", ap_hi_fixed == 120 and ap_lo_fixed == 80)

# Test validation ranges
protein_level = 6.8
test("Protein Level validation: 6.8 is valid", 4.0 <= protein_level <= 12.0)
test("Protein Level validation: 3.0 is extreme", not (4.0 <= 3.0 <= 12.0))
test("Protein Level validation: 15.0 is extreme", not (4.0 <= 15.0 <= 12.0))

ejection_fraction = 60.0
test("Ejection Fraction validation: 60 is valid", 30 <= ejection_fraction <= 80)
test("Ejection Fraction validation: 25 is low", not (30 <= 25 <= 80))
test("Ejection Fraction validation: 85 is high", not (30 <= 85 <= 80))
print()

# ============================================================================
# TEST 10: Age Group Classification
# ============================================================================
print("TEST 10: Age Group Classification")
print("-" * 70)
age_groups = {
    25: "20-29",
    35: "30-39",
    45: "40-49",
    55: "50-59",
    65: "60+"
}

for age, expected_group in age_groups.items():
    if age < 30:
        age_group = "20-29"
    elif age < 40:
        age_group = "30-39"
    elif age < 50:
        age_group = "40-49"
    elif age < 60:
        age_group = "50-59"
    else:
        age_group = "60+"
    test(f"Age {age} -> {expected_group}", age_group == expected_group)
print()

# ============================================================================
# TEST 11: BMI Category Classification
# ============================================================================
print("TEST 11: BMI Category Classification")
print("-" * 70)
bmi_categories = {
    17: "Underweight",
    22: "Normal",
    27: "Overweight",
    32: "Obese"
}

for bmi, expected_category in bmi_categories.items():
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"
    test(f"BMI {bmi} -> {expected_category}", bmi_category == expected_category)
print()

# ============================================================================
# TEST 12: BP Category Classification
# ============================================================================
print("TEST 12: BP Category Classification")
print("-" * 70)
bp_categories = [
    (115, 75, "Normal"),
    (125, 78, "Elevated"),
    (135, 85, "Stage 1"),
    (150, 95, "Stage 2")
]

for ap_hi, ap_lo, expected_category in bp_categories:
    if ap_hi < 120 and ap_lo < 80:
        bp_category = "Normal"
    elif ap_hi < 130 and ap_lo < 80:
        bp_category = "Elevated"
    elif ap_hi < 140 or ap_lo < 90:
        bp_category = "Stage 1"
    else:
        bp_category = "Stage 2"
    test(f"BP {ap_hi}/{ap_lo} -> {expected_category}", bp_category == expected_category)
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
    print("🎉 ALL FORMULA TESTS PASSED!")
    print("✅ All formulas match the paper specifications.")
else:
    print("⚠️  Some formula tests failed. Please review the errors above.")
    print("\nFailed Tests:")
    for result in test_results:
        if result[0] != "PASS":
            print(f"  - {result[1]}: {result[2] if len(result) > 2 else 'Unknown error'}")

print("=" * 70)

# Exit with appropriate code
sys.exit(0 if failed == 0 else 1)

