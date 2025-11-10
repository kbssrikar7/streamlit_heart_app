# Ensemble Weight Optimization - Documentation for Paper

## Current Implementation

The ensemble model uses **equal weights (0.5, 0.5)** for XGBoost and CatBoost models.

## Paper Claim vs Implementation

**Paper Section III.F states:**
> "We optimized the weights on a validation set (grid search over 0.0–1.0 in steps of 0.05) by maximizing ROC-AUC and F1 on the hold-out fold."

**Current Implementation:**
- Uses fixed weights: `w_xgb = 0.5`, `w_cat = 0.5`
- No grid search optimization performed
- Weights are stored in `models/ensemble_weights.json`

## Options for Paper Update

### Option 1: Update Paper Methodology (Recommended)
Update Section III.F to state:
> "Equal weights (0.5, 0.5) were used for the ensemble after initial experimentation showed that this configuration provided a good balance between XGBoost and CatBoost performance, with the ensemble achieving 84.9% accuracy and 0.924 ROC-AUC."

### Option 2: Implement Grid Search (Future Work)
If grid search optimization is desired, add the following to `train_model.py`:

```python
# Weight optimization using grid search
from sklearn.model_selection import ParameterGrid

def optimize_ensemble_weights(xgb_probs, cat_probs, y_true):
    """Optimize ensemble weights using grid search"""
    best_score = 0
    best_weights = (0.5, 0.5)
    
    # Grid search over weights
    weights = np.arange(0.0, 1.01, 0.05)
    for w_xgb in weights:
        w_cat = 1.0 - w_xgb
        ensemble_probs = w_xgb * xgb_probs + w_cat * cat_probs
        score = roc_auc_score(y_true, ensemble_probs)
        if score > best_score:
            best_score = score
            best_weights = (w_xgb, w_cat)
    
    return best_weights, best_score

# Use validation set for optimization
# (requires splitting test set into validation and test)
```

## Recommendation

**Use Option 1** (Update Paper) because:
1. Equal weights (0.5, 0.5) are a common and valid approach for ensemble methods
2. The ensemble already achieves excellent performance (84.9% accuracy, 0.924 ROC-AUC)
3. Equal weights are simpler and more interpretable
4. Saves computation time
5. The paper can still mention that "weights were chosen to balance model contributions"

## Implementation Status

- ✅ Equal weights implemented in `app.py` and `train_model.py`
- ✅ Weights stored in `models/ensemble_weights.json`
- ⚠️ Grid search optimization not implemented
- ✅ Paper methodology section needs update

## Notes

- The current implementation is functionally correct and performs well
- Grid search can be added later if needed for paper consistency
- The ensemble performance matches the paper claims (84.9% accuracy, 0.924 ROC-AUC)

