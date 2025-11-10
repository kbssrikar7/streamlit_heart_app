# Action Items: Paper Implementation Gaps

## 🚨 CRITICAL FIXES REQUIRED (Must Address Before Submission)

### 1. Hybrid Dual-Threshold Risk Mapping
**Status**: ❌ Missing  
**Location**: Paper Section IV.E, Figures 7-8  
**Issue**: Paper describes two mapping strategies (A & B) but implementation only uses single threshold (0.5)

**What to do**:
- Implement Mapping A: Moderate risk (30-70%) → Classify as "At-Risk" (higher recall)
- Implement Mapping B: Moderate risk (30-70%) → Classify as "Safe" (higher precision)
- Add toggle/selector in Streamlit app to switch between mappings
- Update prediction logic in `app.py` to support both mappings

**Files to modify**: `app.py` (prediction logic)

---

### 2. Ensemble Weight Optimization
**Status**: ❌ Missing  
**Location**: Paper Section III.F  
**Issue**: Paper claims grid search optimization, but weights are hardcoded to 0.5-0.5

**What to do**:
- Implement grid search over weights (0.0 to 1.0, steps of 0.05)
- Optimize on validation set using ROC-AUC and F1 score
- Save optimized weights to `ensemble_weights.json`
- **OR** Update paper to state that equal weights (50-50) were used after experimentation

**Files to modify**: `train_model.py` (add weight optimization function)  
**Alternative**: Update paper methodology section

---

### 3. Lifestyle Score Formula Mismatch
**Status**: ⚠️ Discrepancy  
**Location**: Paper Section III.C vs `app.py` line 197-208

**Paper Formula**: `Lifestyle = active - (smoke + alco)` (scaled)  
**Implementation**: `lifestyle_score = smoke + alco + (1 - active)`

**What to do**:
- **Option A**: Update implementation to match paper formula
- **Option B**: Update paper to match implementation (document the actual formula used)

**Files to modify**: `app.py` (line 197-208) OR paper Section III.C

---

### 4. Risk Age Formula Mismatch
**Status**: ⚠️ Discrepancy  
**Location**: Paper Section III.C vs `app.py` line 280

**Paper Formula**: `Risk_Age = age_years + BMI/5 + 2*(cholesterol > 1) + (gluc > 1)`  
**Implementation**: `Risk_Age = age_years + (health_risk_score * 5)`

**What to do**:
- Align formulas - choose one and update the other
- If keeping paper formula, update `app.py` line 280
- If keeping implementation, update paper Section III.C

**Files to modify**: `app.py` (line 280) OR paper Section III.C

---

## 📊 MEDIUM PRIORITY (Important for Paper Completeness)

### 5. Add SHAP Explanations to Streamlit App
**Status**: ❌ Missing  
**Location**: Paper Section III.H, Section IV.D  
**Issue**: SHAP analysis exists in notebook but not in deployed app

**What to do**:
- Install `shap` library in `requirements.txt`
- Add SHAP explanation generation in `app.py` after prediction
- Display SHAP force plot or waterfall plot for individual predictions
- Show top contributing features with their impact

**Files to modify**: 
- `requirements.txt` (add `shap`)
- `app.py` (add SHAP explanation section)

---

### 6. Add Calibration Analysis
**Status**: ❌ Missing  
**Location**: Paper Section III.G  
**Issue**: Paper mentions calibration plots and Brier scores but not implemented

**What to do**:
- Generate calibration plots (reliability diagrams) using `sklearn.calibration`
- Calculate Brier scores for all models
- Add calibration analysis to evaluation script
- Include in results section of paper

**Files to create/modify**: 
- Create `evaluate_calibration.py` script
- Add calibration plots to notebook or results

---

### 7. Add Fairness/Per-Group Metrics
**Status**: ❌ Missing  
**Location**: Paper Section III.G  
**Issue**: Paper mentions per-group metrics but not computed

**What to do**:
- Calculate metrics by age groups (30-39, 40-49, 50-59, 60+)
- Calculate metrics by gender (Male/Female)
- Calculate metrics by hypertension flag (Yes/No)
- Create fairness report table

**Files to create/modify**: 
- Create `fairness_analysis.py` script
- Add to evaluation notebook

---

### 8. Add PR-AUC Metric
**Status**: ❌ Missing  
**Location**: Paper Section III.G  
**Issue**: Paper mentions PR-AUC but only ROC-AUC is computed

**What to do**:
- Import `average_precision_score` from sklearn
- Calculate PR-AUC for all models
- Add to metrics table
- Include in paper results

**Files to modify**: 
- `train_model.py` (add PR-AUC calculation)
- Notebook evaluation cells

---

### 9. Implement Reason String Generation
**Status**: ❌ Missing  
**Location**: Paper Section III.C, Section III.I  
**Issue**: Paper mentions "Reason" string for interpretability but not generated in app

**What to do**:
- Create function to generate human-readable reason string
- Include key risk factors (e.g., "High BP, Low EF, Obese")
- Display in Streamlit app after prediction
- Format: "High BP, Elevated Cholesterol, Inactive Lifestyle"

**Files to modify**: `app.py` (add reason generation function)

---

## 🔧 LOW PRIORITY (Nice to Have)

### 10. Complete Training Script
**Status**: ⚠️ Incomplete  
**Location**: `train_model.py`  
**Issue**: Only trains XGBoost and CatBoost, not all 5 models

**What to do**:
- Add Logistic Regression training
- Add Random Forest training  
- Add LightGBM training
- Generate complete metrics comparison table
- Save all models

**Files to modify**: `train_model.py`

---

### 11. Update Documentation
**Status**: ⚠️ Needs Updates  
**Location**: `README.md`  
**Issue**: Documentation doesn't match all paper features

**What to do**:
- Update README with hybrid mapping feature
- Document SHAP explanations
- Add calibration and fairness metrics
- Update feature list to match paper exactly

**Files to modify**: `README.md`

---

## ✅ QUICK WINS (Easy Fixes)

1. **Update README.md** - Add missing features mentioned in paper
2. **Add PR-AUC** - Simple metric addition (5 minutes)
3. **Fix Lifestyle Score** - Update formula in `app.py` (10 minutes)
4. **Fix Risk Age** - Update formula in `app.py` (10 minutes)
5. **Add Reason String** - Simple string generation (30 minutes)

---

## 📋 IMPLEMENTATION CHECKLIST

### Before Paper Submission:
- [ ] Fix hybrid mapping implementation
- [ ] Resolve weight optimization discrepancy (implement OR update paper)
- [ ] Fix Lifestyle Score formula mismatch
- [ ] Fix Risk Age formula mismatch
- [ ] Add SHAP to Streamlit app
- [ ] Add calibration analysis
- [ ] Add fairness metrics
- [ ] Add PR-AUC metric
- [ ] Implement reason string generation
- [ ] Update paper methodology section with actual formulas used
- [ ] Update paper results section with all promised metrics

### For Code Quality:
- [ ] Complete training script with all models
- [ ] Add unit tests for feature engineering
- [ ] Document all formula choices
- [ ] Ensure reproducibility (random seeds, etc.)

---

## 🎯 RECOMMENDED ORDER OF IMPLEMENTATION

1. **Fix Formula Discrepancies** (Quick wins)
   - Lifestyle Score
   - Risk Age
   - Reason String

2. **Add Missing Metrics** (Medium effort)
   - PR-AUC
   - Calibration plots
   - Fairness metrics

3. **Implement Key Features** (Higher effort)
   - Hybrid mappings
   - SHAP in app
   - Weight optimization

4. **Polish & Documentation** (Final steps)
   - Complete training script
   - Update README
   - Update paper

---

## 📝 NOTES

- Most performance metrics **match** the paper, so the core implementation is correct
- The main gaps are in **evaluation completeness** and **interpretability features**
- The notebook has most of the analysis; need to integrate into training script and app
- Paper claims are mostly accurate, but some features need to be implemented or documented differently

---

**Priority**: Fix critical issues first, then medium priority items  
**Timeline**: Critical fixes should be done before paper submission  
**Impact**: High - These discrepancies could be caught during paper review

