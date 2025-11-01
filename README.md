# Heart Attack Risk Predictor - Streamlit App

A Streamlit web application for predicting heart attack risk using an ensemble machine learning model (XGBoost + CatBoost).

## Features

- **Ensemble Model**: Combines XGBoost and CatBoost models for accurate predictions
- **User-Friendly Interface**: Intuitive form-based input for patient information
- **Real-time Predictions**: Instant risk assessment with probability scores
- **Detailed Analysis**: Shows predictions from individual models and ensemble
- **Responsive Design**: Clean and modern UI

## Model Performance

- **Accuracy**: ~85%
- **ROC-AUC**: ~92%
- **Precision**: ~85%
- **Recall**: ~84%

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd streamlit_heart_app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

Before running the Streamlit app, you need to train the models:

1. Place your dataset `cardio_train_extended.csv` in the project root directory
2. Run the training script:
```bash
python train_model.py
```

This will:
- Train XGBoost and CatBoost models
- Save the models and preprocessor to the `models/` directory
- Generate feature information files

## Running the App Locally

Once the models are trained, run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Deployment on Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Connect your GitHub repository
5. Select the `streamlit_heart_app` directory
6. Set main file to `app.py`
7. Deploy!

**Important**: Make sure to include the `models/` directory in your GitHub repository (or use a cloud storage solution).

## Usage

1. Fill in the patient information form:
   - Demographics (Gender, Height, Weight)
   - Blood Pressure measurements
   - Medical History (Cholesterol, Glucose, Smoking, Alcohol)
   - Physical Activity status
   - Additional health metrics

2. Click "🔮 Predict Heart Attack Risk"

3. View the results:
   - Risk probability percentage
   - Risk level (High/Low)
   - Detailed model breakdown
   - Recommendations based on risk level

## Project Structure

```
streamlit_heart_app/
├── app.py                 # Main Streamlit application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/               # Generated after training
│   ├── preprocessor.joblib
│   ├── xgb_model.joblib
│   ├── cat_model.joblib
│   ├── feature_info.json
│   └── ensemble_weights.json
└── cardio_train_extended.csv  # Dataset (not included in repo)
```

## Features Used by the Model

The model uses the following features:

### Numeric Features:
- Gender, Height, Weight
- Blood Pressure (Systolic, Diastolic)
- BMI, Pulse Pressure
- Age, Lifestyle Score
- Protein Level, Ejection Fraction
- And more...

### Categorical Features:
- Age Group (Young/Middle/Senior)
- BMI Category (Underweight/Normal/Overweight/Obese)
- BP Category (Normal/Elevated/High Stage 1/High Stage 2)
- Risk Level (Low/Medium/High)

## Model Architecture

- **Preprocessing**: StandardScaler for numeric features, OneHotEncoder for categorical
- **Models**: XGBoost and CatBoost (ensemble with 50-50 weights)
- **Evaluation**: Stratified train-test split (80-20)

## Disclaimer

⚠️ **Important**: This tool is for educational and demonstration purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

## License

This project is provided as-is for educational purposes.
