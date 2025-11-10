"""
Streamlit App for Heart Attack Risk Prediction
Based on ensemble model (XGBoost + CatBoost)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import matplotlib
# Set backend before importing pyplot - critical for Streamlit Cloud
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# Ensure matplotlib is properly configured for Streamlit
plt.ioff()  # Turn off interactive mode

# Configure matplotlib fonts for better readability
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Verdana', 'Liberation Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP library not available. Install with: pip install shap")

# Page configuration
st.set_page_config(
    page_title="Predicting Heart Attack Risk: An Ensemble Modeling Approach",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF1744;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin-top: 2rem;
    }
    .risk-high {
        color: #FF1744;
        font-size: 2rem;
        font-weight: bold;
    }
    .risk-low {
        color: #4CAF50;
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and preprocessor
@st.cache_resource
def load_models():
    """Load models and preprocessor (cached for performance)"""
    models_dir = Path("models")
    
    try:
        preprocessor = joblib.load(models_dir / "preprocessor.joblib")
        xgb_model = joblib.load(models_dir / "xgb_model.joblib")
        cat_model = joblib.load(models_dir / "cat_model.joblib")
        
        with open(models_dir / "feature_info.json", 'r') as f:
            feature_info = json.load(f)
        
        with open(models_dir / "ensemble_weights.json", 'r') as f:
            ensemble_weights = json.load(f)
        
        return preprocessor, xgb_model, cat_model, feature_info, ensemble_weights
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please run train_model.py first to generate the model files.")
        return None, None, None, None, None

# Load models
preprocessor, xgb_model, cat_model, feature_info, ensemble_weights = load_models()

# Main title
st.markdown('<h1 class="main-header">Predicting Heart Attack Risk: An Ensemble Modeling Approach</h1>', unsafe_allow_html=True)
st.markdown("---")

if preprocessor is None:
    st.stop()

# Sidebar for model info and settings
with st.sidebar:
    st.header("📊 Model Information")
    st.markdown("""
    **Ensemble Model:**
    - XGBoost (50% weight)
    - CatBoost (50% weight)
    
    **Performance:**
    - Accuracy: ~84%
    - Recall: ~84%
    - ROC-AUC: ~92%
    """)
    
    st.markdown("---")
    st.header("⚙️ Risk Classification Strategy")
    
    st.markdown("""
    **Choose how to classify patients based on risk probability:**
    
    **When to use each strategy:**
    - **Standard (50%)**: Balanced approach for general use
    - **Mapping A (30%)**: Use for **screening** - catch more cases early
    - **Mapping B (70%)**: Use for **conservative** - reduce false alarms
    """)
    
    mapping_strategy = st.radio(
        "Select Strategy:",
        options=["Standard", "Mapping A (High Recall)", "Mapping B (High Precision)"],
        help="""
        **Standard (50% threshold)**:
        - Balanced precision and recall
        - Best for general clinical use
        - Risk ≥50% = At-Risk
        
        **Mapping A (30% threshold)**:
        - Higher recall (catches more cases)
        - Use for: Population screening, early detection
        - Risk ≥30% = At-Risk
        - May have more false positives
        
        **Mapping B (70% threshold)**:
        - Higher precision (fewer false alarms)
        - Use for: Conservative approach, reducing unnecessary tests
        - Risk ≥70% = At-Risk
        - May miss some moderate-risk cases
        """
    )
    
    # Show explanation based on selection
    if mapping_strategy == "Standard":
        st.info("📌 **Standard Mode**: Balanced approach. Risk ≥50% classified as At-Risk.")
    elif mapping_strategy == "Mapping A (High Recall)":
        st.warning("📌 **Mapping A (Screening Mode)**: More sensitive. Risk ≥30% classified as At-Risk. Best for catching cases early.")
    else:
        st.success("📌 **Mapping B (Conservative Mode)**: More specific. Risk ≥70% classified as At-Risk. Reduces false alarms.")

# Input form with all features
st.header("📝 Patient Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    height = st.number_input("Height (cm)", min_value=100, max_value=220, value=170, step=1)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
    
    # Calculate BMI with category
    bmi = weight / ((height / 100) ** 2) if height > 0 else 0
    if bmi < 18.5:
        bmi_status = "⚠️ Underweight"
        bmi_color = "inverse"
    elif bmi < 25:
        bmi_status = "✅ Normal"
        bmi_color = "normal"
    elif bmi < 30:
        bmi_status = "⚠️ Overweight"
        bmi_color = "normal"
    else:
        bmi_status = "🔴 Obese"
        bmi_color = "inverse"
    
    st.metric("BMI", f"{bmi:.2f}", delta=bmi_status, delta_color=bmi_color, 
              help="Body Mass Index - Healthy range: 18.5-24.9")

with col2:
    st.subheader("Blood Pressure")
    ap_hi = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=120, step=1)
    ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80, step=1)
    
    # Calculate BP_diff and category
    bp_diff = ap_hi - ap_lo
    
    # BP Status
    if ap_hi < 120 and ap_lo < 80:
        bp_status = "✅ Normal"
        bp_color = "normal"
    elif ap_hi < 130 and ap_lo < 80:
        bp_status = "⚠️ Elevated"
        bp_color = "normal"
    elif ap_hi < 140 or ap_lo < 90:
        bp_status = "🔴 Stage 1"
        bp_color = "inverse"
    else:
        bp_status = "🚨 Stage 2"
        bp_color = "inverse"
    
    st.metric("Pulse Pressure", f"{bp_diff} mmHg", delta=bp_status, delta_color=bp_color,
              help="Normal BP: <120/80 mmHg")

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Medical History")
    cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], 
                              format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}.get(x))
    gluc = st.selectbox("Glucose Level", options=[1, 2, 3],
                       format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}.get(x))
    smoke = st.radio("Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    alco = st.radio("Alcohol Consumption", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)

with col4:
    st.subheader("Activity & Derived Features")
    active = st.radio("Physical Activity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    
    # Age in days (for model compatibility - we'll use age_years for display)
    age_years = st.number_input("Age (years)", min_value=20, max_value=100, value=50, step=1)
    age_days = age_years * 365  # Convert to days for model
    
    # Derived features (set defaults based on common values)
    systolic_pressure = ap_hi
    map_value = ap_lo + (bp_diff / 3)  # Mean Arterial Pressure approximation
    pulse_pressure_ratio = bp_diff / ap_hi if ap_hi > 0 else 0

# Additional derived features (using defaults/calculations)
st.markdown("---")
st.subheader("Additional Health Metrics")

col5, col6, col7 = st.columns(3)

with col5:
    protein_level = st.number_input("Protein Level", min_value=0.0, max_value=200.0, value=14.0, step=0.1)
    
with col6:
    ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
    
with col7:
    # Calculate Lifestyle Score based on paper formula: Lifestyle = active - (smoke + alco)
    # Paper formula: active - (smoke + alco) 
    # Range: -2 (worst: inactive + smoke + alcohol) to 1 (best: active, no smoke, no alcohol)
    # For model compatibility, we normalize to 0-3 range (matching training data format)
    lifestyle_score_raw = active - (smoke + alco)  # Paper formula: range -2 to 1
    # Normalize to 0-3 for model compatibility (inverted: higher = worse)
    # Maps: -2→3, -1→2, 0→1, 1→0 (worst to best lifestyle)
    lifestyle_score = 3 - (lifestyle_score_raw + 2) if lifestyle_score_raw >= -2 else 3
    
    risk_factors = []
    if smoke == 1:
        risk_factors.append("Smoking")
    if alco == 1:
        risk_factors.append("Alcohol")
    if active == 0:
        risk_factors.append("Inactive")
    
    # Display calculated lifestyle score with risk indicator
    if lifestyle_score == 0:
        score_label = "✅ Low Risk"
        delta_color = "normal"
    elif lifestyle_score == 1:
        score_label = "⚠️ Moderate Risk"
        delta_color = "normal"
    elif lifestyle_score == 2:
        score_label = "🔴 High Risk"
        delta_color = "inverse"
    else:
        score_label = "🚨 Very High Risk"
        delta_color = "inverse"
    
    st.metric(
        "Lifestyle Risk Score", 
        f"{lifestyle_score}/3 - {score_label}",
        help=f"Formula: active - (smoke + alco), scaled for display. Risk factors: {', '.join(risk_factors) if risk_factors else 'None'}"
    )
    if risk_factors:
        st.caption(f"⚠️ Risk factors: {', '.join(risk_factors)}")

# Calculate additional derived features
obesity_flag = 1 if bmi >= 30 else 0
hypertension_flag = 1 if ap_hi >= 140 or ap_lo >= 90 else 0
health_risk_score = lifestyle_score + obesity_flag + hypertension_flag
smoker_alcoholic = 1 if (smoke == 1 or alco == 1) else 0

# Age group and BMI category (categorical features needed by model)
# Must match dataset values exactly: '30-39', '40-49', '50-59', '60+'
if age_years < 30:
    age_group = "20-29"  # Fallback for ages < 30
elif age_years < 40:
    age_group = "30-39"
elif age_years < 50:
    age_group = "40-49"
elif age_years < 60:
    age_group = "50-59"
else:
    age_group = "60+"

if bmi < 18.5:
    bmi_category = "Underweight"
elif bmi < 25:
    bmi_category = "Normal"
elif bmi < 30:
    bmi_category = "Overweight"
else:
    bmi_category = "Obese"

# BP Category
# Must match dataset values exactly: 'Normal', 'Elevated', 'Stage 1', 'Stage 2'
if ap_hi < 120 and ap_lo < 80:
    bp_category = "Normal"
elif ap_hi < 130 and ap_lo < 80:
    bp_category = "Elevated"
elif ap_hi < 140 or ap_lo < 90:
    bp_category = "Stage 1"  # Fixed: was "High Stage 1"
else:
    bp_category = "Stage 2"  # Fixed: was "High Stage 2"

# Risk Level
if health_risk_score <= 2:
    risk_level = "Low"
elif health_risk_score <= 4:
    risk_level = "Medium"
else:
    risk_level = "High"

# Risk Age (derived) - Paper formula: age_years + BMI/5 + 2*(cholesterol > 1) + (gluc > 1)
risk_age = age_years + (bmi / 5) + (2 * (1 if cholesterol > 1 else 0)) + (1 if gluc > 1 else 0)

# Create feature dictionary
feature_dict = {
    'gender': gender,
    'height': height,
    'weight': weight,
    'ap_hi': ap_hi,
    'ap_lo': ap_lo,
    'cholesterol': cholesterol,
    'gluc': gluc,
    'smoke': smoke,
    'alco': alco,
    'active': active,
    'BMI': bmi,
    'BP_diff': bp_diff,
    'Systolic_Pressure': systolic_pressure,
    'age_years': age_years,
    'Age_Group': age_group,
    'Lifestyle_Score': lifestyle_score,
    'Obesity_Flag': obesity_flag,
    'Hypertension_Flag': hypertension_flag,
    'Health_Risk_Score': health_risk_score,
    'Pulse_Pressure_Ratio': pulse_pressure_ratio,
    'MAP': map_value,
    'BMI_Category': bmi_category,
    'Smoker_Alcoholic': smoker_alcoholic,
    'BP_Category': bp_category,
    'Risk_Age': risk_age,
    'Risk_Level': risk_level,
    'Protein_Level': protein_level,
    'Ejection_Fraction': ejection_fraction
}

# Health Summary Card (before prediction)
st.markdown("---")
st.subheader("📊 Health Summary")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    if obesity_flag == 1:
        st.error("🔴 Obesity Risk")
    else:
        st.success("✅ Healthy Weight")

with summary_col2:
    if hypertension_flag == 1:
        st.error("🔴 Hypertension")
    else:
        st.success("✅ Normal BP")

with summary_col3:
    if lifestyle_score >= 2:
        st.error(f"🔴 High Lifestyle Risk ({lifestyle_score}/3)")
    elif lifestyle_score == 1:
        st.warning(f"⚠️ Moderate Risk ({lifestyle_score}/3)")
    else:
        st.success("✅ Low Risk (0/3)")

with summary_col4:
    if cholesterol == 3 or gluc == 3:
        st.error("🔴 Elevated Levels")
    elif cholesterol == 2 or gluc == 2:
        st.warning("⚠️ Above Normal")
    else:
        st.success("✅ Normal Levels")

# Prediction button
st.markdown("---")
predict_button = st.button("🔮 Predict Heart Attack Risk", type="primary", use_container_width=True)

if predict_button:
    try:
        # Get expected feature order from training (num_cols + cat_cols)
        expected_features = feature_info.get('num_cols', []) + feature_info.get('cat_cols', [])
        
        # Create DataFrame with features in the exact order expected by the model
        input_df = pd.DataFrame([feature_dict])
        
        # Reorder columns to match training data order
        # Ensure all expected columns are present
        for feat in expected_features:
            if feat not in input_df.columns:
                st.warning(f"Missing feature: {feat}")
        
        # Reorder to match training order
        available_features = [f for f in expected_features if f in input_df.columns]
        input_df = input_df[available_features]
        
        # Transform using preprocessor
        X_processed = preprocessor.transform(input_df)
        
        # Get predictions from both models
        xgb_prob = xgb_model.predict_proba(X_processed)[0, 1]
        cat_prob = cat_model.predict_proba(X_processed)[0, 1]
        
        # Ensemble prediction (weighted average)
        w_xgb = ensemble_weights.get('w_xgb', 0.5)
        w_cat = ensemble_weights.get('w_cat', 0.5)
        ensemble_prob = w_xgb * xgb_prob + w_cat * cat_prob
        
        risk_percentage = ensemble_prob * 100
        
        # Hybrid Dual-Threshold Risk Mapping (Paper Section IV.E)
        # Mapping A: Moderate (30-70%) → At-Risk (higher recall for screening)
        # Mapping B: Moderate (30-70%) → Safe (higher precision, fewer false alarms)
        # Standard: Uses 0.5 threshold
        if mapping_strategy == "Mapping A (High Recall)":
            # Moderate risk treated as at-risk (threshold at 30%)
            prediction = 1 if risk_percentage >= 30 else 0
            mapping_description = "Moderate risk (30-70%) classified as 'At-Risk' for early detection"
        elif mapping_strategy == "Mapping B (High Precision)":
            # Moderate risk treated as safe (threshold at 70%)
            prediction = 1 if risk_percentage >= 70 else 0
            mapping_description = "Moderate risk (30-70%) classified as 'Safe' to reduce false alarms"
        else:
            # Standard threshold at 50%
            prediction = 1 if ensemble_prob >= 0.5 else 0
            mapping_description = "Standard threshold (50%) for balanced precision and recall"
        
        # Generate Reason String for interpretability (Paper Section III.C)
        reason_parts = []
        if ap_hi >= 140 or ap_lo >= 90:
            reason_parts.append("High BP")
        if bmi >= 30:
            reason_parts.append("Obese")
        elif bmi >= 25:
            reason_parts.append("Overweight")
        if cholesterol >= 3:
            reason_parts.append("High Cholesterol")
        elif cholesterol >= 2:
            reason_parts.append("Elevated Cholesterol")
        if gluc >= 3:
            reason_parts.append("High Glucose")
        elif gluc >= 2:
            reason_parts.append("Elevated Glucose")
        if ejection_fraction < 50:
            reason_parts.append("Low EF")
        elif ejection_fraction < 60:
            reason_parts.append("Reduced EF")
        if smoke == 1:
            reason_parts.append("Smoking")
        if alco == 1:
            reason_parts.append("Alcohol")
        if active == 0:
            reason_parts.append("Inactive")
        if age_years >= 60:
            reason_parts.append("Advanced Age")
        
        reason_string = ", ".join(reason_parts) if reason_parts else "Low risk profile"
        
        # Display results
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        # Show mapping strategy info with better explanation
        if mapping_strategy == "Mapping A (High Recall)":
            st.info(f"📌 **Strategy**: {mapping_strategy}\n\n{mapping_description}\n\n💡 **Use Case**: Screening mode - catches more cases early. Recommended for population screening.")
        elif mapping_strategy == "Mapping B (High Precision)":
            st.success(f"📌 **Strategy**: {mapping_strategy}\n\n{mapping_description}\n\n💡 **Use Case**: Conservative mode - reduces false alarms. Recommended when you want high confidence.")
        else:
            st.info(f"📌 **Strategy**: {mapping_strategy}\n\n{mapping_description}\n\n💡 **Use Case**: Balanced approach - good for general clinical use.")
        
        # Main result with visual indicator
        if prediction == 1:
            st.error(f"⚠️ **HIGH RISK DETECTED** - {risk_percentage:.1f}% probability of heart disease")
        else:
            st.success(f"✅ **LOW RISK** - {risk_percentage:.1f}% probability of heart disease")
        
        # Display Reason String
        st.markdown(f"**🔍 Risk Factors**: {reason_string}")
        
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.metric("Risk Probability", f"{risk_percentage:.2f}%", 
                     delta=f"{'High' if risk_percentage >= 70 else 'Moderate' if risk_percentage >= 50 else 'Low'} Risk",
                     delta_color="inverse" if risk_percentage >= 70 else "normal")
        
        with col_result2:
            # Visual risk level
            if risk_percentage >= 70:
                risk_level = "🚨 Very High"
            elif risk_percentage >= 50:
                risk_level = "🔴 High"
            elif risk_percentage >= 30:
                risk_level = "⚠️ Moderate"
            else:
                risk_level = "✅ Low"
            st.metric("Risk Level", risk_level)
        
        with col_result3:
            st.metric("Prediction", "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
                     delta="Consult Doctor" if prediction == 1 else "Continue Monitoring",
                     delta_color="inverse" if prediction == 1 else "normal")
        
        # Enhanced progress bar with color coding
        risk_bar_color = "#FF1744" if risk_percentage >= 70 else "#FF9800" if risk_percentage >= 50 else "#4CAF50"
        st.markdown(f"""
        <div style="background-color: #f0f0f0; border-radius: 5px; padding: 10px; margin: 10px 0;">
            <div style="background-color: {risk_bar_color}; width: {risk_percentage}%; height: 30px; border-radius: 5px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                {risk_percentage:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed breakdown with visual bars
        with st.expander("📊 Model Details & Breakdown"):
            col_model1, col_model2, col_model3 = st.columns(3)
            with col_model1:
                st.write("**XGBoost Model**")
                st.progress(float(xgb_prob))  # Convert to Python float
                st.caption(f"{float(xgb_prob)*100:.2f}% risk")
            with col_model2:
                st.write("**CatBoost Model**")
                st.progress(float(cat_prob))  # Convert to Python float
                st.caption(f"{float(cat_prob)*100:.2f}% risk")
            with col_model3:
                st.write("**Ensemble (Final)**")
                st.progress(float(ensemble_prob))  # Convert to Python float
                st.caption(f"{float(ensemble_prob)*100:.2f}% risk")
            
            # Display actual ensemble weights dynamically
            w_xgb_pct = int(w_xgb * 100)
            w_cat_pct = int(w_cat * 100)
            st.info(f"💡 **Ensemble Method**: Weighted average ({w_xgb_pct}% XGBoost + {w_cat_pct}% CatBoost) for more reliable predictions")
        
        # SHAP Explanations with Graph Visualization (Paper Section III.H, IV.D)
        if SHAP_AVAILABLE:
            with st.expander("🔍 SHAP Explanation (Model Interpretability)"):
                try:
                    st.write("**Feature Importance for This Prediction:**")
                    # Use CatBoost model for SHAP (as mentioned in paper)
                    explainer = shap.TreeExplainer(cat_model)
                    shap_values = explainer.shap_values(X_processed)
                    
                    # Get feature names
                    feature_names = feature_info.get('processed_feature_names', [f'Feature_{i}' for i in range(X_processed.shape[1])])
                    
                    # For binary classification, use class 1 (positive class)
                    if isinstance(shap_values, list):
                        shap_vals = shap_values[1]  # Positive class
                    else:
                        shap_vals = shap_values
                    
                    # Get SHAP values for this single prediction
                    shap_vals_single = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
                    
                    # Create DataFrame with feature names and SHAP values
                    shap_df = pd.DataFrame({
                        'Feature': feature_names[:len(shap_vals_single)],
                        'SHAP Value': shap_vals_single
                    })
                    shap_df['Abs_SHAP'] = np.abs(shap_df['SHAP Value'])
                    shap_df = shap_df.sort_values('Abs_SHAP', ascending=False).head(10)
                    
                    # Create bar chart visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Color code: Dark Red for positive (increases risk), Dark Blue for negative (decreases risk)
                    colors = ['#C62828' if x > 0 else '#1565C0' for x in shap_df['SHAP Value']]
                    
                    # Calculate spacing parameters based on SHAP value range
                    max_abs_value = shap_df['SHAP Value'].abs().max()
                    min_value = shap_df['SHAP Value'].min()
                    max_value = shap_df['SHAP Value'].max()
                    
                    # Calculate offset for text labels (12% of max absolute value for better spacing)
                    text_offset = max_abs_value * 0.12
                    
                    # Calculate x-axis limits with padding for text labels (25% padding)
                    xlim_padding = max_abs_value * 0.25
                    x_min = min_value - xlim_padding
                    x_max = max_value + xlim_padding
                    
                    # Create horizontal bar chart
                    bars = ax.barh(shap_df['Feature'], shap_df['SHAP Value'], 
                                  color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                    
                    # Set x-axis limits to accommodate text labels
                    ax.set_xlim(x_min, x_max)
                    
                    # Add vertical line at zero
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
                    
                    # Customize axes
                    ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11, fontweight='bold')
                    ax.set_title('Top 10 Feature Contributions to Heart Disease Risk', 
                               fontsize=13, fontweight='bold', pad=15)
                    ax.grid(axis='x', alpha=0.3, linestyle='--')
                    ax.set_facecolor('#f8f9fa')
                    
                    # Add legend
                    legend_elements = [
                        Patch(facecolor='#C62828', alpha=0.7, label='Increases Risk'),
                        Patch(facecolor='#1565C0', alpha=0.7, label='Decreases Risk')
                    ]
                    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
                    
                    # Add value labels on bars with proper spacing to avoid overlap
                    for i, (idx, row) in enumerate(shap_df.iterrows()):
                        value = row['SHAP Value']
                        if value > 0:
                            # For positive values: place text to the right of the bar with offset
                            ax.text(value + text_offset, i, 
                                   f'{value:.3f}', 
                                   va='center', ha='left',
                                   fontsize=9, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                            alpha=0.8, edgecolor='gray', linewidth=0.5))
                        else:
                            # For negative values: place text to the left of the bar with offset
                            ax.text(value - text_offset, i, 
                                   f'{value:.3f}', 
                                   va='center', ha='right',
                                   fontsize=9, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                            alpha=0.8, edgecolor='gray', linewidth=0.5))
                    
                    plt.tight_layout()
                    
                    # Display the graph - use clear_figure=False to ensure it displays
                    try:
                        st.pyplot(fig, clear_figure=False)
                    except Exception as plot_error:
                        # Fallback: save to buffer and display
                        import io
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                        buf.seek(0)
                        st.image(buf, use_container_width=True)
                        st.warning(f"Graph displayed using fallback method. Original error: {str(plot_error)}")
                    
                    plt.close(fig)
                    
                    # Text summary below graph
                    st.markdown("---")
                    st.markdown("### 📊 Detailed Feature Contributions:")
                    
                    col_shap1, col_shap2 = st.columns(2)
                    
                    with col_shap1:
                        increases_risk = shap_df[shap_df['SHAP Value'] > 0]
                        if len(increases_risk) > 0:
                            st.markdown("**🔴 Features that Increase Risk:**")
                            for idx, row in increases_risk.iterrows():
                                st.write(f"• **{row['Feature']}**: +{row['SHAP Value']:.4f}")
                        else:
                            st.info("No features increase risk for this prediction.")
                    
                    with col_shap2:
                        decreases_risk = shap_df[shap_df['SHAP Value'] < 0]
                        if len(decreases_risk) > 0:
                            st.markdown("**🔵 Features that Decrease Risk:**")
                            for idx, row in decreases_risk.iterrows():
                                st.write(f"• **{row['Feature']}**: {row['SHAP Value']:.4f}")
                        else:
                            st.info("No features decrease risk for this prediction.")
                    
                    st.caption("💡 **Positive SHAP values** (red bars) increase the predicted risk of heart disease.")
                    st.caption("💡 **Negative SHAP values** (blue bars) decrease the predicted risk of heart disease.")
                    st.caption("📈 The bar chart shows how each feature contributes to the final prediction. Longer bars indicate greater impact.")
                    
                except Exception as e:
                    st.error(f"⚠️ Error generating SHAP graph: {str(e)}")
                    import traceback
                    with st.expander("🔍 Error Details (for debugging)"):
                        st.code(traceback.format_exc())
                    
                    # Fallback: Show text-only SHAP values if graph fails
                    st.info("📊 **Fallback: Feature Contributions (Text Only)**")
                    try:
                        # Try to get SHAP values even if graph failed
                        explainer = shap.TreeExplainer(cat_model)
                        shap_values = explainer.shap_values(X_processed)
                        
                        if isinstance(shap_values, list):
                            shap_vals = shap_values[1]
                        else:
                            shap_vals = shap_values
                        
                        shap_vals_single = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
                        feature_names = feature_info.get('processed_feature_names', [f'Feature_{i}' for i in range(X_processed.shape[1])])
                        
                        shap_df_fallback = pd.DataFrame({
                            'Feature': feature_names[:len(shap_vals_single)],
                            'SHAP Value': shap_vals_single
                        })
                        shap_df_fallback['Abs_SHAP'] = np.abs(shap_df_fallback['SHAP Value'])
                        shap_df_fallback = shap_df_fallback.sort_values('Abs_SHAP', ascending=False).head(10)
                        
                        for idx, row in shap_df_fallback.iterrows():
                            icon = "🔴" if row['SHAP Value'] > 0 else "🔵"
                            st.write(f"{icon} **{row['Feature']}**: {row['SHAP Value']:.4f}")
                    except:
                        st.warning("Could not generate SHAP values. Please check that SHAP library is properly installed.")
        else:
            with st.expander("🔍 SHAP Explanation (Model Interpretability)"):
                st.info("SHAP library not installed. Install with: `pip install shap` to enable feature explanations.")
        
        # Recommendations
        st.markdown("---")
        if prediction == 1 or risk_percentage > 70:
            st.warning("⚠️ **High Risk Detected!** Please consult with a healthcare professional immediately.")
            st.info("""
            **Recommendations:**
            - Schedule an appointment with a cardiologist
            - Monitor blood pressure regularly
            - Maintain a healthy diet and exercise routine
            - Avoid smoking and limit alcohol consumption
            - Follow up with regular health checkups
            """)
        elif risk_percentage > 50:
            st.warning("⚠️ **Moderate Risk** - Consider consulting a healthcare professional.")
        else:
            st.success("✅ **Low Risk** - Continue maintaining a healthy lifestyle!")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
    <p>Always seek the advice of qualified health providers with any questions you may have regarding a medical condition.</p>
</div>
""", unsafe_allow_html=True)

