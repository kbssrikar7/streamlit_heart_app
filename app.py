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

# Page configuration
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
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
st.markdown('<h1 class="main-header">❤️ Heart Attack Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

if preprocessor is None:
    st.stop()

# Sidebar for model info
with st.sidebar:
    st.header("📊 Model Information")
    st.markdown("""
    **Ensemble Model:**
    - XGBoost (50% weight)
    - CatBoost (50% weight)
    
    **Performance:**
    - Accuracy: ~85%
    - ROC-AUC: ~92%
    
    **Note:** This is a prediction tool, not a medical diagnosis.
    Always consult healthcare professionals for medical advice.
    """)

# Input form with all features
st.header("📝 Patient Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    height = st.number_input("Height (cm)", min_value=100, max_value=220, value=170, step=1)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
    
    # Calculate BMI
    bmi = weight / ((height / 100) ** 2) if height > 0 else 0
    st.metric("BMI", f"{bmi:.2f}", help="Body Mass Index")

with col2:
    st.subheader("Blood Pressure")
    ap_hi = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=120, step=1)
    ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80, step=1)
    
    # Calculate BP_diff
    bp_diff = ap_hi - ap_lo
    st.metric("Pulse Pressure", f"{bp_diff} mmHg")

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
    # Calculate Lifestyle Score automatically based on lifestyle factors
    # Lifestyle Score = sum of risk factors (smoking, alcohol, inactivity)
    lifestyle_score = 0
    if smoke == 1:
        lifestyle_score += 1  # Smoking adds to lifestyle risk
    if alco == 1:
        lifestyle_score += 1  # Alcohol adds to lifestyle risk
    if active == 0:
        lifestyle_score += 1  # Inactivity adds to lifestyle risk
    
    # Display calculated lifestyle score
    st.metric("Lifestyle Score", lifestyle_score, 
              help="Auto-calculated: +1 for smoking, +1 for alcohol, +1 for inactivity (Range: 0-3)")

# Calculate additional derived features
obesity_flag = 1 if bmi >= 30 else 0
hypertension_flag = 1 if ap_hi >= 140 or ap_lo >= 90 else 0
health_risk_score = lifestyle_score + obesity_flag + hypertension_flag
smoker_alcoholic = 1 if (smoke == 1 or alco == 1) else 0

# Age group and BMI category (categorical features needed by model)
if age_years < 40:
    age_group = "Young"
elif age_years < 60:
    age_group = "Middle"
else:
    age_group = "Senior"

if bmi < 18.5:
    bmi_category = "Underweight"
elif bmi < 25:
    bmi_category = "Normal"
elif bmi < 30:
    bmi_category = "Overweight"
else:
    bmi_category = "Obese"

# BP Category
if ap_hi < 120 and ap_lo < 80:
    bp_category = "Normal"
elif ap_hi < 130 and ap_lo < 80:
    bp_category = "Elevated"
elif ap_hi < 140 or ap_lo < 90:
    bp_category = "High Stage 1"
else:
    bp_category = "High Stage 2"

# Risk Level
if health_risk_score <= 2:
    risk_level = "Low"
elif health_risk_score <= 4:
    risk_level = "Medium"
else:
    risk_level = "High"

# Risk Age (derived)
risk_age = age_years + (health_risk_score * 5)

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
        
        # Binary prediction
        prediction = 1 if ensemble_prob >= 0.5 else 0
        risk_percentage = ensemble_prob * 100
        
        # Display results
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.metric("Risk Probability", f"{risk_percentage:.2f}%")
        
        with col_result2:
            if prediction == 1:
                st.markdown('<p class="risk-high">⚠️ HIGH RISK</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="risk-low">✅ LOW RISK</p>', unsafe_allow_html=True)
        
        with col_result3:
            st.metric("Prediction", "Heart Disease Detected" if prediction == 1 else "No Heart Disease")
        
        # Progress bar for risk
        st.progress(ensemble_prob)
        
        # Detailed breakdown
        with st.expander("📊 Model Details"):
            col_model1, col_model2 = st.columns(2)
            with col_model1:
                st.write("**XGBoost Prediction:**")
                st.write(f"Risk Probability: {xgb_prob*100:.2f}%")
            with col_model2:
                st.write("**CatBoost Prediction:**")
                st.write(f"Risk Probability: {cat_prob*100:.2f}%")
            st.write(f"**Ensemble Weighted Average:** {ensemble_prob*100:.2f}%")
        
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

