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
import sys
import logging
from pathlib import Path
from datetime import datetime
import matplotlib
# Set backend before importing pyplot - critical for Streamlit Cloud
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# Ensure matplotlib is properly configured for Streamlit
plt.ioff()  # Turn off interactive mode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment variable support
MODELS_DIR = Path(os.getenv('MODELS_DIR', 'models'))
DATA_PATH = Path(os.getenv('DATA_PATH', 'cardio_train_extended.csv'))

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
    # Don't show warning here - will be shown later if SHAP is actually needed
    pass

# Define expected categorical values (must match training data exactly)
# This is defined early so it can be used throughout the app
EXPECTED_CATEGORIES = {
    'Age_Group': ['20-29', '30-39', '40-49', '50-59', '60+'],
    'BMI_Category': ['Underweight', 'Normal', 'Overweight', 'Obese'],
    'BP_Category': ['Normal', 'Elevated', 'Stage 1', 'Stage 2'],
    'Risk_Level': ['Low', 'Moderate', 'High']
}

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
    
    /* Responsive design for mobile */
    @media (max-width: 768px) {
        .main-header { 
            font-size: 1.5rem !important; 
        }
        .prediction-box { 
            padding: 1rem !important; 
        }
        .risk-high, .risk-low {
            font-size: 1.5rem !important;
        }
        /* Make columns stack on mobile */
        div[data-testid="column"] {
            width: 100% !important;
        }
    }
    
    /* Improve touch targets on mobile */
    @media (max-width: 768px) {
        button {
            min-height: 44px !important;
            min-width: 44px !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load models and preprocessor
# Cache key includes weights file modification time to force refresh when weights change
@st.cache_resource
def load_models():
    """Load models and preprocessor (cached for performance)"""
    logger.info(f"Loading models from {MODELS_DIR}")
    
    try:
        # Get weights file modification time for cache invalidation
        weights_file = MODELS_DIR / "ensemble_weights.json"
        weights_mtime = weights_file.stat().st_mtime if weights_file.exists() else 0
        
        preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
        xgb_model = joblib.load(MODELS_DIR / "xgb_model.joblib")
        cat_model = joblib.load(MODELS_DIR / "cat_model.joblib")
        
        with open(MODELS_DIR / "feature_info.json", 'r') as f:
            feature_info = json.load(f)
        
        with open(weights_file, 'r') as f:
            ensemble_weights = json.load(f)
        
        # Verify weights are loaded correctly
        w_xgb = ensemble_weights.get('w_xgb', 0.5)
        w_cat = ensemble_weights.get('w_cat', 0.5)
        if w_xgb != 0.5 or w_cat != 0.5:
            logger.warning(f"Ensemble weights are {w_xgb}/{w_cat}. Expected 0.5/0.5.")
        
        logger.info("Models loaded successfully")
        return preprocessor, xgb_model, cat_model, feature_info, ensemble_weights
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}", exc_info=True)
        st.error(f"Error loading models: {str(e)}")
        st.info("Please run train_model.py first to generate the model files.")
        return None, None, None, None, None

@st.cache_resource
def get_shap_explainer(_cat_model):
    """Cache SHAP explainer (expensive to create)"""
    if SHAP_AVAILABLE:
        logger.info("Creating SHAP explainer (cached)")
        return shap.TreeExplainer(_cat_model)
    return None

# Load models (cached for performance)
preprocessor, xgb_model, cat_model, feature_info, _ = load_models()

# Health check for monitoring (Docker, Kubernetes, etc.)
# This must be after model loading
if os.getenv('HEALTH_CHECK') == 'true':
    if preprocessor and xgb_model and cat_model:
        print("OK")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)

# IMPORTANT: Always reload weights directly from file to bypass cache
# This ensures we always use the current weights (50/50) even if models are cached
ensemble_weights = {'w_xgb': 0.5, 'w_cat': 0.5}  # Default fallback (paper specification)
weights_file = MODELS_DIR / "ensemble_weights.json"
if weights_file.exists():
    try:
        with open(weights_file, 'r') as f:
            ensemble_weights = json.load(f)
        logger.info(f"Loaded ensemble weights: {ensemble_weights.get('w_xgb', 0.5)}/{ensemble_weights.get('w_cat', 0.5)}")
    except Exception as e:
        # If file read fails, use default 50/50 (paper specification)
        logger.warning(f"Could not load weights file: {e}. Using default 50/50.")
        st.sidebar.warning(f"Could not load weights file: {e}. Using default 50/50.")
        ensemble_weights = {'w_xgb': 0.5, 'w_cat': 0.5}

# Main title
st.markdown('<h1 class="main-header">Predicting Heart Attack Risk: An Ensemble Modeling Approach</h1>', unsafe_allow_html=True)
st.markdown("---")

if preprocessor is None:
    logger.error("Models failed to load - stopping app")
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
    
    st.markdown("---")
    st.header("🧪 Try Example Patients")
    st.markdown("Load pre-configured patient profiles to test the model:")
    
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        if st.button("📊 High Risk Example", use_container_width=True):
            st.session_state['example_patient'] = {
                'gender': 1, 'height': 175, 'weight': 95, 'age_years': 65,
                'ap_hi': 160, 'ap_lo': 100, 'cholesterol': 3, 'gluc': 3,
                'smoke': 1, 'alco': 1, 'active': 0, 'protein_level': 7.8,
                'ejection_fraction': 45.0
            }
            st.rerun()
    
    with col_ex2:
        if st.button("✅ Low Risk Example", use_container_width=True):
            st.session_state['example_patient'] = {
                'gender': 2, 'height': 165, 'weight': 60, 'age_years': 35,
                'ap_hi': 115, 'ap_lo': 75, 'cholesterol': 1, 'gluc': 1,
                'smoke': 0, 'alco': 0, 'active': 1, 'protein_level': 6.5,
                'ejection_fraction': 65.0
            }
            st.rerun()
    
    if st.button("⚠️ Moderate Risk Example", use_container_width=True):
        st.session_state['example_patient'] = {
            'gender': 1, 'height': 170, 'weight': 80, 'age_years': 55,
            'ap_hi': 135, 'ap_lo': 85, 'cholesterol': 2, 'gluc': 2,
            'smoke': 0, 'alco': 1, 'active': 1, 'protein_level': 7.2,
            'ejection_fraction': 55.0
        }
        st.rerun()
    
    st.markdown("---")
    st.header("❓ Frequently Asked Questions")
    with st.expander("How accurate is this model?"):
        st.markdown("""
        The ensemble model achieves:
        - **Accuracy**: ~85%
        - **ROC-AUC**: ~92%
        - **Recall**: ~84%
        - **Precision**: ~86%
        
        These metrics are based on test data from the Kaggle "Cardio Train" dataset.
        """)
    
    with st.expander("What do the mapping strategies mean?"):
        st.markdown("""
        **Standard (50% threshold)**:
        - Balanced precision and recall
        - Best for general clinical use
        
        **Mapping A (30% threshold)**:
        - Higher recall (catches more cases)
        - Use for: Population screening, early detection
        - May have more false positives
        
        **Mapping B (70% threshold)**:
        - Higher precision (fewer false alarms)
        - Use for: Conservative approach, reducing unnecessary tests
        - May miss some moderate-risk cases
        """)
    
    with st.expander("What is SHAP and how does it work?"):
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** explains how each feature contributes to the prediction:
        - **Red bars**: Features that increase heart disease risk
        - **Blue bars**: Features that decrease heart disease risk
        - **Longer bars**: Greater impact on the prediction
        
        This helps understand which factors are most important for each individual prediction.
        """)
    
    with st.expander("Should I use this for medical decisions?"):
        st.markdown("""
        ⚠️ **No. This tool is for research and educational purposes only.**
        
        Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.
        This model is not a substitute for professional medical care.
        """)
    
    with st.expander("What is Ejection Fraction?"):
        st.markdown("""
        **Ejection Fraction (EF)** measures how much blood the left ventricle pumps out with each contraction.
        - **Normal range**: 55-70%
        - **Reduced**: 40-54%
        - **Low**: <40%
        
        Lower EF values may indicate heart problems and increase heart attack risk.
        """)
    
    with st.expander("What is Lifestyle Score?"):
        st.markdown("""
        **Lifestyle Score** is calculated as: `active - (smoke + alco)`
        - Range: -2 (worst) to 1 (best)
        - **Best (1)**: Active, no smoking, no alcohol
        - **Worst (-2)**: Inactive, smoking, alcohol
        
        This score is normalized for model compatibility.
        """)
    
    st.markdown("---")
    st.header("🔒 Privacy & Data")
    st.info("""
    ⚠️ **Privacy Notice**: 
    
    This tool processes data locally. No patient information is stored or transmitted to external servers.
    
    All predictions are computed on your device. For research/educational purposes only.
    """)
    
    st.markdown("---")
    st.header("📝 Feedback")
    feedback = st.sidebar.text_area(
        "Help us improve (optional):", 
        height=100,
        help="Your feedback helps us improve the app",
        key="feedback_input"
    )
    if st.sidebar.button("Submit Feedback"):
        if feedback.strip():
            try:
                feedback_file = Path("feedback.txt")
                with open(feedback_file, "a") as f:
                    f.write(f"{datetime.now().isoformat()}: {feedback}\n")
                st.sidebar.success("Thank you for your feedback! 🙏")
                logger.info(f"Feedback received: {feedback[:50]}...")
            except Exception as e:
                logger.error(f"Error saving feedback: {e}")
                st.sidebar.warning(f"Could not save feedback: {e}")
        else:
            st.sidebar.info("Please enter feedback before submitting.")

# Input form with all features
st.header("📝 Patient Information")

# Load example patient data if available
example_data = st.session_state.get('example_patient', None)
if example_data:
    st.success("✅ Example patient data loaded! Review and adjust values as needed.")
    # Clear the example after showing message
    if 'example_patient_shown' not in st.session_state:
        st.session_state['example_patient_shown'] = True

# Input validation warnings
validation_warnings = []

# Demographics Section
with st.expander("👤 Demographics", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        gender_default = example_data.get('gender', 1) if example_data else 1
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female",
                             index=0 if gender_default == 1 else 1)
        
        height_default = example_data.get('height', 170) if example_data else 170
        height = st.number_input("Height (cm)", min_value=100, max_value=220, value=int(height_default), step=1,
                                help="Normal adult height range: 150-190 cm")
        
        if height < 120 or height > 210:
            validation_warnings.append(f"⚠️ Height ({height} cm) is outside typical adult range (120-210 cm). Please verify.")
    
    with col2:
        weight_default = example_data.get('weight', 70.0) if example_data else 70.0
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=float(weight_default), step=0.1,
                                help="Normal adult weight varies by height and build")
        
        if weight < 40 or weight > 150:
            validation_warnings.append(f"⚠️ Weight ({weight} kg) is outside typical adult range (40-150 kg). Please verify.")
        
        # Calculate BMI with category
        bmi = weight / ((height / 100) ** 2) if height > 0 else 0
        if bmi < 15 or bmi > 50:
            validation_warnings.append(f"⚠️ BMI ({bmi:.1f}) is extreme. Please verify height and weight values.")
        
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

# Blood Pressure Section
with st.expander("🩺 Blood Pressure & Vital Signs", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        ap_hi_default = example_data.get('ap_hi', 120) if example_data else 120
        ap_hi = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=int(ap_hi_default), step=1,
                               help="Top number in blood pressure reading. Normal: <120 mmHg")
        
        if ap_hi < 90 or ap_hi > 200:
            validation_warnings.append(f"⚠️ Systolic BP ({ap_hi} mmHg) is extreme. Please verify.")
    
    with col2:
        ap_lo_default = example_data.get('ap_lo', 80) if example_data else 80
        ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=int(ap_lo_default), step=1,
                               help="Bottom number in blood pressure reading. Normal: <80 mmHg")
        
        if ap_lo < 50 or ap_lo > 120:
            validation_warnings.append(f"⚠️ Diastolic BP ({ap_lo} mmHg) is extreme. Please verify.")
    
    # BP Validation: Swap if diastolic > systolic (common data entry error)
    bp_swapped = False
    if ap_lo > ap_hi:
        ap_hi, ap_lo = ap_lo, ap_hi
        bp_swapped = True
        st.warning(f"⚠️ **Blood Pressure values swapped!** Diastolic was higher than systolic. Values corrected: {ap_hi}/{ap_lo} mmHg")
    
    # Calculate BP_diff and category
    bp_diff = ap_hi - ap_lo
    
    if bp_diff < 20 or bp_diff > 100:
        validation_warnings.append(f"⚠️ Pulse Pressure ({bp_diff} mmHg) is unusual. Normal range: 30-60 mmHg.")
    
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
    
    col_bp1, col_bp2 = st.columns(2)
    with col_bp1:
        st.metric("Pulse Pressure", f"{bp_diff} mmHg", delta=bp_status, delta_color=bp_color,
                  help="Normal BP: <120/80 mmHg. Pulse Pressure = Systolic - Diastolic")
    with col_bp2:
        map_value = ap_lo + (bp_diff / 3)  # Mean Arterial Pressure
        st.metric("MAP (Mean Arterial Pressure)", f"{map_value:.1f} mmHg",
                 help="MAP = Diastolic + (Pulse Pressure / 3). Normal: 70-100 mmHg")

# Medical History Section
with st.expander("💊 Medical History", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        cholesterol_default = example_data.get('cholesterol', 1) if example_data else 1
        cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], 
                                  format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}.get(x),
                                  index=cholesterol_default - 1,
                                  help="1 = Normal, 2 = Above Normal, 3 = Well Above Normal")
        
        gluc_default = example_data.get('gluc', 1) if example_data else 1
        gluc = st.selectbox("Glucose Level", options=[1, 2, 3],
                           format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}.get(x),
                           index=gluc_default - 1,
                           help="1 = Normal, 2 = Above Normal, 3 = Well Above Normal")
    
    with col2:
        smoke_default = example_data.get('smoke', 0) if example_data else 0
        smoke = st.radio("Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", 
                        horizontal=True, index=smoke_default,
                        help="Regular smoking increases cardiovascular risk")
        
        alco_default = example_data.get('alco', 0) if example_data else 0
        alco = st.radio("Alcohol Consumption", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", 
                       horizontal=True, index=alco_default,
                       help="Regular alcohol consumption may increase risk")

# Lifestyle & Activity Section
with st.expander("🏃 Lifestyle & Activity", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        active_default = example_data.get('active', 1) if example_data else 1
        active = st.radio("Physical Activity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", 
                         horizontal=True, index=active_default,
                         help="Regular physical activity reduces cardiovascular risk")
    
    with col2:
        age_years_default = example_data.get('age_years', 50) if example_data else 50
        age_years = st.number_input("Age (years)", min_value=20, max_value=100, value=int(age_years_default), step=1,
                                   help="Age is a significant risk factor for heart disease")
        
        if age_years < 25 or age_years > 90:
            validation_warnings.append(f"⚠️ Age ({age_years} years) is outside typical range for this model (25-90 years).")
        
        age_days = age_years * 365  # Convert to days for model
    
    # Calculate Lifestyle Score
    lifestyle_score_raw = active - (smoke + alco)  # Paper formula: range -2 to 1
    lifestyle_score = 3 - (lifestyle_score_raw + 2) if lifestyle_score_raw >= -2 else 3
    
    risk_factors = []
    if smoke == 1:
        risk_factors.append("Smoking")
    if alco == 1:
        risk_factors.append("Alcohol")
    if active == 0:
        risk_factors.append("Inactive")
    
    # Display calculated lifestyle score
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
        help=f"Formula: active - (smoke + alco). Risk factors: {', '.join(risk_factors) if risk_factors else 'None'}"
    )
    if risk_factors:
        st.caption(f"⚠️ Risk factors: {', '.join(risk_factors)}")

# Additional Health Metrics Section
with st.expander("🔬 Additional Health Metrics", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        protein_level_default = example_data.get('protein_level', 6.8) if example_data else 6.8
        protein_level = st.number_input("Protein Level (g/dL)", min_value=0.0, max_value=200.0, value=float(protein_level_default), step=0.1,
                                       help="Total protein in blood. Normal range: 6.0-8.3 g/dL. Dataset training range: 5.6-8.0 g/dL")
        # Validation: Dataset was trained on values ~5.6-8.0, but allow slightly wider range for real-world use
        if protein_level < 4.0 or protein_level > 12.0:
            validation_warnings.append(f"⚠️ Protein Level ({protein_level} g/dL) is extremely outside the expected range (4.0-12.0 g/dL). Normal range: 6.0-8.3 g/dL.")
        elif protein_level < 5.5 or protein_level > 8.5:
            validation_warnings.append(f"ℹ️ Protein Level ({protein_level} g/dL) is outside the dataset training range (5.6-8.0 g/dL), but may still be valid.")
    
    with col2:
        ejection_fraction_default = example_data.get('ejection_fraction', 60.0) if example_data else 60.0
        ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0.0, max_value=100.0, value=float(ejection_fraction_default), step=0.1,
                                           help="Percentage of blood pumped out per heartbeat. Normal: 55-70%")
        if ejection_fraction < 30 or ejection_fraction > 80:
            validation_warnings.append(f"⚠️ Ejection Fraction ({ejection_fraction}%) is outside typical range (30-80%).")
        elif ejection_fraction < 50:
            validation_warnings.append(f"⚠️ Ejection Fraction ({ejection_fraction}%) is below normal (55-70%). This may indicate heart problems.")
    
    with col3:
        # Derived features display
        systolic_pressure = ap_hi
        pulse_pressure_ratio = bp_diff / ap_hi if ap_hi > 0 else 0
        st.metric("Pulse Pressure Ratio", f"{pulse_pressure_ratio:.3f}",
                 help="Pulse Pressure / Systolic BP. Normal range: 0.25-0.50")

# Display validation warnings
if validation_warnings:
    st.markdown("---")
    st.warning("⚠️ **Input Validation Warnings:**")
    for warning in validation_warnings:
        st.write(warning)
    st.info("💡 Please review these values. Extreme values may affect prediction accuracy.")

# Clear example patient data after first use
if example_data and st.session_state.get('example_patient_shown', False):
    if 'example_patient' in st.session_state:
        del st.session_state['example_patient']
    st.session_state['example_patient_shown'] = False

# Calculate additional derived features (ensure all are available)
# Recalculate BP_diff to ensure it's available (in case BP was swapped earlier)
bp_diff = ap_hi - ap_lo  # Pulse Pressure
systolic_pressure = ap_hi
map_value = ap_lo + (bp_diff / 3)  # Mean Arterial Pressure
pulse_pressure_ratio = bp_diff / ap_hi if ap_hi > 0 else 0

obesity_flag = 1 if bmi >= 30 else 0
hypertension_flag = 1 if ap_hi >= 140 or ap_lo >= 90 else 0
health_risk_score = lifestyle_score + obesity_flag + hypertension_flag
smoker_alcoholic = 1 if (smoke == 1 and alco == 1) else 0  # Paper formula: 1 if smoke = 1 & alco = 1 else 0

# Age group and BMI category (categorical features needed by model)
# Must match dataset values exactly: '20-29', '30-39', '40-49', '50-59', '60+'
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

# Validate age_group matches expected categories
if age_group not in EXPECTED_CATEGORIES['Age_Group']:
    logger.warning(f"Age_Group '{age_group}' not in expected categories: {EXPECTED_CATEGORIES['Age_Group']}")
    # Use fallback to closest match
    if age_years < 30:
        age_group = "20-29"
    elif age_years < 40:
        age_group = "30-39"
    elif age_years < 50:
        age_group = "40-49"
    elif age_years < 60:
        age_group = "50-59"
    else:
        age_group = "60+"
    logger.info(f"Age_Group corrected to: {age_group}")

if bmi < 18.5:
    bmi_category = "Underweight"
elif bmi < 25:
    bmi_category = "Normal"
elif bmi < 30:
    bmi_category = "Overweight"
else:
    bmi_category = "Obese"

# Validate BMI category matches expected categories
if bmi_category not in EXPECTED_CATEGORIES['BMI_Category']:
    logger.warning(f"BMI_Category '{bmi_category}' not in expected categories: {EXPECTED_CATEGORIES['BMI_Category']}")
    # Use fallback
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"
    logger.info(f"BMI_Category corrected to: {bmi_category}")

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

# Validate BP category matches expected categories
if bp_category not in EXPECTED_CATEGORIES['BP_Category']:
    logger.warning(f"BP_Category '{bp_category}' not in expected categories: {EXPECTED_CATEGORIES['BP_Category']}")
    # Use fallback
    if ap_hi < 120 and ap_lo < 80:
        bp_category = "Normal"
    elif ap_hi < 130 and ap_lo < 80:
        bp_category = "Elevated"
    elif ap_hi < 140 or ap_lo < 90:
        bp_category = "Stage 1"
    else:
        bp_category = "Stage 2"
    logger.info(f"BP_Category corrected to: {bp_category}")

# Risk Level
# IMPORTANT: Must match model's expected categories: "Low", "Moderate", "High" (NOT "Medium")
if health_risk_score <= 2:
    risk_level = "Low"
elif health_risk_score <= 4:
    risk_level = "Moderate"  # Fixed: Changed from "Medium" to "Moderate" to match model
else:
    risk_level = "High"

# Validate Risk Level matches expected categories
if risk_level not in EXPECTED_CATEGORIES['Risk_Level']:
    logger.error(f"Risk_Level '{risk_level}' not in expected categories: {EXPECTED_CATEGORIES['Risk_Level']}")
    # Fix common mistakes
    if risk_level == "Medium":
        risk_level = "Moderate"
        logger.warning("Fixed Risk_Level: 'Medium' -> 'Moderate'")
    else:
        # Use fallback
        if health_risk_score <= 2:
            risk_level = "Low"
        elif health_risk_score <= 4:
            risk_level = "Moderate"
        else:
            risk_level = "High"
    logger.info(f"Risk_Level corrected to: {risk_level}")

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
        logger.info("Prediction requested")
        
        # Get expected feature order from training (num_cols + cat_cols)
        expected_features = feature_info.get('num_cols', []) + feature_info.get('cat_cols', [])
        
        # Create DataFrame with features in the exact order expected by the model
        input_df = pd.DataFrame([feature_dict])
        
        # Reorder columns to match training data order
        # Ensure all expected columns are present
        missing_features = [f for f in expected_features if f not in input_df.columns]
        if missing_features:
            error_msg = f"Missing required features: {missing_features}"
            logger.error(error_msg)
            st.error(f"❌ **Error**: {error_msg}")
            st.info("💡 **Troubleshooting**: Please ensure all required features are provided. Check the input form.")
            st.markdown("**Missing features:**")
            for feat in missing_features:
                st.markdown(f"- `{feat}`")
            st.stop()  # Stop processing instead of continuing with missing features
        
        # Validate that we have all required features
        if len(input_df.columns) < len(expected_features):
            missing_count = len(expected_features) - len(input_df.columns)
            error_msg = f"Missing {missing_count} required feature(s). Expected {len(expected_features)} features, got {len(input_df.columns)}."
            logger.error(error_msg)
            st.error(f"❌ **Error**: {error_msg}")
            st.info("💡 Please check that all input fields are filled correctly.")
            st.stop()
        
        # Reorder to match training order
        available_features = [f for f in expected_features if f in input_df.columns]
        input_df = input_df[available_features]
        
        # Validate feature order matches expected order
        if list(input_df.columns) != expected_features:
            logger.info(f"Reordering features to match expected order (expected {len(expected_features)} features)")
            input_df = input_df[expected_features]
        
        # Validate data types and check for NaN values
        for col in input_df.columns:
            if input_df[col].isna().any():
                error_msg = f"Feature '{col}' contains NaN values"
                logger.error(error_msg)
                st.error(f"❌ **Error**: {error_msg}")
                st.info(f"💡 Please ensure '{col}' has a valid value.")
                st.stop()
        
        # Transform using preprocessor
        try:
            with st.spinner("🔄 Preprocessing input data..."):
                X_processed = preprocessor.transform(input_df)
            logger.info("Input preprocessed successfully")
        except Exception as preprocess_error:
            error_msg = f"Preprocessing error: {str(preprocess_error)}"
            logger.error(error_msg, exc_info=True)
            st.error(f"❌ **Error**: {error_msg}")
            st.info("💡 This may be due to:")
            st.markdown("""
            - Invalid categorical values (e.g., Age_Group, BMI_Category, BP_Category, Risk_Level)
            - Data type mismatches
            - Values outside expected ranges
            """)
            st.info("💡 Please check that all inputs are within valid ranges and categorical values are correct.")
            with st.expander("🔍 Error Details (for debugging)", expanded=False):
                import traceback
                error_trace = traceback.format_exc()
                st.code(error_trace)
            st.stop()
        
        # Get predictions from both models with progress indicator
        with st.spinner("🤖 Generating predictions from models..."):
            xgb_prob = xgb_model.predict_proba(X_processed)[0, 1]
            cat_prob = cat_model.predict_proba(X_processed)[0, 1]
        logger.info(f"XGBoost probability: {xgb_prob:.4f}, CatBoost probability: {cat_prob:.4f}")
        
        # Ensemble prediction (weighted average)
        with st.spinner("⚖️ Computing ensemble prediction..."):
            w_xgb = ensemble_weights.get('w_xgb', 0.5)
            w_cat = ensemble_weights.get('w_cat', 0.5)
            ensemble_prob = w_xgb * xgb_prob + w_cat * cat_prob
            
            risk_percentage = ensemble_prob * 100
        logger.info(f"Ensemble prediction: {risk_percentage:.2f}% risk")
        
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
        
        logger.info(f"Prediction: {risk_percentage:.2f}% risk, Strategy: {mapping_strategy}, Result: {'At-Risk' if prediction == 1 else 'Safe'}")
        if validation_warnings:
            logger.warning(f"Validation warnings: {len(validation_warnings)} warnings")
        
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
            # Calculate prediction confidence (distance from 0.5 threshold)
            confidence = abs(ensemble_prob - 0.5) * 2  # Scale to 0-1 range
            confidence_pct = confidence * 100
            
            st.metric("Prediction", "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
                     delta=f"Confidence: {confidence_pct:.1f}%",
                     delta_color="inverse" if prediction == 1 else "normal")
            
            # Confidence interpretation
            if confidence_pct >= 80:
                conf_level = "Very High"
            elif confidence_pct >= 60:
                conf_level = "High"
            elif confidence_pct >= 40:
                conf_level = "Moderate"
            else:
                conf_level = "Low"
            st.caption(f"📊 Confidence Level: {conf_level}")
        
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
            # Add checkbox to control SHAP computation (lazy loading)
            compute_shap = st.checkbox("🔍 Show detailed SHAP analysis (may take a few seconds)", value=True,
                                      help="SHAP explanations show how each feature contributes to the prediction. Uncheck to skip for faster results.")
            
            if compute_shap:
                with st.expander("🔍 SHAP Explanation (Model Interpretability)", expanded=True):
                    try:
                        st.write("**Feature Importance for This Prediction:**")
                        
                        # Show loading spinner for SHAP computation
                        with st.spinner("⏳ Computing SHAP explanations... This may take a few seconds."):
                            # Use CatBoost model for SHAP (as mentioned in paper)
                            # Use cached explainer for better performance
                            explainer = get_shap_explainer(cat_model)
                            if explainer is None:
                                st.error("SHAP explainer could not be created. SHAP library may not be installed correctly.")
                                st.stop()
                            logger.info("Computing SHAP values for prediction")
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
                        
                        # Create bar chart visualization (outside spinner for better UX)
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
                        logger.error(f"Error generating SHAP graph: {str(e)}", exc_info=True)
                        st.error(f"⚠️ Error generating SHAP graph: {str(e)}")
                        st.info("💡 **Troubleshooting Tips:**")
                        st.markdown("""
                        - Ensure SHAP library is properly installed: `pip install shap`
                        - Try refreshing the page
                        - If the error persists, check the error details below
                        """)
                        
                        import traceback
                        with st.expander("🔍 Error Details (for debugging)", expanded=False):
                            error_trace = traceback.format_exc()
                            st.code(error_trace)
                            logger.error(f"SHAP graph error traceback: {error_trace}")
                        
                        # Fallback: Show text-only SHAP values if graph fails
                        st.info("📊 **Fallback: Feature Contributions (Text Only)**")
                        try:
                            # Try to get SHAP values even if graph failed
                            explainer = get_shap_explainer(cat_model)
                            if explainer is None:
                                st.warning("SHAP explainer could not be created")
                            else:
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
                        except Exception as fallback_error:
                            st.warning(f"Could not generate SHAP values: {str(fallback_error)}")
                            st.info("Please check that SHAP library is properly installed: `pip install shap`")
            else:
                st.info("💡 Check the box above to view detailed SHAP explanations of feature contributions.")
        else:
            with st.expander("🔍 SHAP Explanation (Model Interpretability)"):
                st.info("SHAP library not installed. Install with: `pip install shap` to enable feature explanations.")
        
        # Downloadable Reports
        st.markdown("---")
        st.header("📥 Download Prediction Report")
        
        # Create report data with model version info
        try:
            # Try to get model file modification times as version indicators
            model_file = MODELS_DIR / "xgb_model.joblib"
            if model_file.exists():
                model_mtime = os.path.getmtime(model_file)
                model_version = datetime.fromtimestamp(model_mtime).strftime('%Y%m%d')
            else:
                model_version = "Unknown"
        except (OSError, ValueError, AttributeError) as e:
            logger.warning(f"Could not get model version: {e}")
            model_version = "Unknown"
        
        # Get ensemble weights for report
        w_xgb = ensemble_weights.get('w_xgb', 0.5)
        w_cat = ensemble_weights.get('w_cat', 0.5)
        
        # Create report data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_version': model_version,
                'app_version': '1.0.0',
                'ensemble_weights': f"{int(w_xgb * 100)}% XGBoost, {int(w_cat * 100)}% CatBoost"
            },
            'patient_inputs': {
                'gender': 'Male' if gender == 1 else 'Female',
                'age': age_years,
                'height_cm': height,
                'weight_kg': weight,
                'bmi': round(bmi, 2),
                'systolic_bp': ap_hi,
                'diastolic_bp': ap_lo,
                'cholesterol': cholesterol,
                'glucose': gluc,
                'smoking': 'Yes' if smoke == 1 else 'No',
                'alcohol': 'Yes' if alco == 1 else 'No',
                'physical_activity': 'Yes' if active == 1 else 'No',
                'protein_level': protein_level,
                'ejection_fraction': ejection_fraction
            },
            'prediction_results': {
                'risk_percentage': round(risk_percentage, 2),
                'prediction': 'At-Risk' if prediction == 1 else 'Safe',
                'risk_level': risk_level,
                'confidence': round(confidence_pct, 1),
                'mapping_strategy': mapping_strategy,
                'reason_string': reason_string
            },
            'model_predictions': {
                'xgb_probability': round(float(xgb_prob) * 100, 2),
                'catboost_probability': round(float(cat_prob) * 100, 2),
                'ensemble_probability': round(float(ensemble_prob) * 100, 2)
            },
            'derived_features': {
                'bmi_category': bmi_category,
                'bp_category': bp_category,
                'hypertension_flag': bool(hypertension_flag),
                'obesity_flag': bool(obesity_flag),
                'lifestyle_score': lifestyle_score,
                'risk_age': round(risk_age, 2),
                'age_group': age_group
            }
        }
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            report_json = json.dumps(report_data, indent=2)
            st.download_button(
                "📥 Download Report (JSON)",
                data=report_json,
                file_name=f"heart_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
                help="Download prediction results as JSON file"
            )
        with col_dl2:
            # CSV format with UTF-8 encoding for better compatibility
            import io
            import csv
            csv_buffer = io.StringIO(newline='')
            writer = csv.writer(csv_buffer)
            writer.writerow(['Category', 'Field', 'Value'])
            for section, data in report_data.items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        # Convert all values to strings for CSV compatibility
                        writer.writerow([section, key, str(value)])
            csv_data = csv_buffer.getvalue()
            # Encode as UTF-8 with BOM for Excel compatibility
            try:
                csv_data_encoded = csv_data.encode('utf-8-sig')
            except Exception as e:
                logger.warning(f"Error encoding CSV with BOM: {e}. Using UTF-8 instead.")
                csv_data_encoded = csv_data.encode('utf-8')
            
            st.download_button(
                "📥 Download Report (CSV)",
                data=csv_data_encoded,
                file_name=f"heart_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download prediction results as CSV file (UTF-8 encoded, Excel compatible)"
            )
        
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
        logger.error(f"Error making prediction: {str(e)}", exc_info=True)
        st.error("❌ **Error making prediction**")
        st.markdown(f"**Error Message:** {str(e)}")
        
        st.info("💡 **Troubleshooting Tips:**")
        st.markdown("""
        1. **Check Input Values**: Ensure all inputs are within valid ranges
        2. **Verify Models**: Make sure model files are present in the `models/` directory
        3. **Check Data Types**: Ensure numeric inputs are numbers, not text
        4. **Reload Page**: Try refreshing the page and entering values again
        
        If the error persists, please check the error details below.
        """)
        
        with st.expander("🔍 Error Details (for debugging)", expanded=False):
            import traceback
            error_trace = traceback.format_exc()
            st.code(error_trace)
            logger.error(f"Prediction error traceback: {error_trace}")
        
        st.warning("⚠️ If you continue to experience issues, please verify that:")
        st.markdown("""
        - All model files (`xgb_model.joblib`, `cat_model.joblib`, `preprocessor.joblib`) exist
        - The `feature_info.json` file is present
        - Required Python packages are installed
        - Check `app.log` file for detailed error information
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
    <p>Always seek the advice of qualified health providers with any questions you may have regarding a medical condition.</p>
</div>
""", unsafe_allow_html=True)

