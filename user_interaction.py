# Description: This script provides a command-line and web-based interface for making predictions using the trained model.
import joblib
import pandas as pd
import argparse
from feature_engineering import engineer_features

MODEL_PATH = "models/task_predictor.joblib"
SCALER_PATH = "models/scaler.joblib"
COLUMNS_PATH = "models/training_columns.joblib"

def process_input(features):
    """Preprocess user input for prediction"""
    # Create DataFrame
    input_df = pd.DataFrame([features], columns=[
        'age', 'time_spent', 'tasks_completed', 'wellness_score'
    ])
    
    # Apply feature engineering
    input_df = engineer_features(input_df)
    
    # One-hot encoding
    input_df = pd.get_dummies(input_df, columns=['age_group'], drop_first=True)
    
    # Align with training columns
    training_columns = joblib.load(COLUMNS_PATH)
    input_df = input_df.reindex(columns=training_columns, fill_value=0)
    
    # Scale features
    scaler = joblib.load(SCALER_PATH)
    return scaler.transform(input_df)

def predict_probability(features):
    """Make prediction using trained model"""
    processed_data = process_input(features)
    model = joblib.load(MODEL_PATH)
    return model.predict_proba(processed_data)[0][1]

def cli_interface():
    """Command-line prediction interface"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--age', type=int, required=True)
    parser.add_argument('--time_spent', type=float, required=True)
    parser.add_argument('--tasks_completed', type=int, required=True)
    parser.add_argument('--wellness_score', type=float, required=True)
    
    args = parser.parse_args()
    proba = predict_probability([
        args.age,
        args.time_spent,
        args.tasks_completed,
        args.wellness_score
    ])
    print(f"Predicted completion likelihood: {proba:.1%}")

def streamlit_interface():
    """Web-based GUI for predictions"""
    try:
        import streamlit as st
    except ImportError:
        return
    
    st.title("Task Completion Predictor")
    st.write("Enter user activity metrics:")
    
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        time_spent = st.number_input("Time spent (hours)", min_value=0.0, value=4.5)
        tasks = st.number_input("Tasks completed", min_value=0, value=5)
        wellness = st.number_input("Wellness score (1-10)", min_value=1.0, 
                                 max_value=10.0, value=7.5)
        
        if st.form_submit_button("Predict"):
            proba = predict_probability([age, time_spent, tasks, wellness])
            st.success(f"Completion Probability: {proba:.1%}")

if __name__ == "__main__":
    try:
        streamlit_interface()
    except:
        cli_interface()