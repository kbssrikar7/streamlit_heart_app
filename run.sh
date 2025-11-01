#!/bin/bash
# Quick start script for Streamlit Heart Attack Risk Predictor

echo "❤️ Heart Attack Risk Predictor - Setup & Run"
echo "=============================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if models directory exists and has models
if [ ! -d "models" ] || [ ! -f "models/xgb_model.joblib" ]; then
    echo ""
    echo "⚠️  Models not found!"
    echo "Please make sure you have:"
    echo "  1. The dataset file: cardio_train_extended.csv"
    echo "  2. Run: python train_model.py"
    echo ""
    read -p "Do you want to train the models now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ ! -f "cardio_train_extended.csv" ]; then
            echo "❌ Error: cardio_train_extended.csv not found!"
            echo "Please place your dataset file in the current directory."
            exit 1
        fi
        echo "Training models..."
        python train_model.py
    else
        echo "Skipping model training."
    fi
fi

# Run Streamlit app
echo ""
echo "Starting Streamlit app..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
streamlit run app.py

