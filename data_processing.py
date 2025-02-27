# Description: This script loads the sample dataset, preprocesses it, and saves the processed data, the scaler, and the column structure for later use.
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from feature_engineering import engineer_features

DATA_PATH = "data/user_activity.csv"
SCALER_PATH = "models/scaler.joblib"
COLUMNS_PATH = "models/training_columns.joblib"

def load_data():
    """Load or generate sample dataset"""
    if not os.path.exists(DATA_PATH):
        os.makedirs("data", exist_ok=True)
        np.random.seed(42)
        data = pd.DataFrame({
            'age': np.random.randint(18, 65, 1000),
            'time_spent': np.abs(np.random.normal(5, 2, 1000)),
            'tasks_completed': np.random.poisson(5, 1000),
            'wellness_score': np.random.uniform(1, 10, 1000),
            'task_completion_likelihood': np.random.choice([0, 1], 1000, p=[0.3, 0.7])
        })
        data.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)

def preprocess_data(df):
    """Full preprocessing pipeline"""
    df = engineer_features(df)
    df = pd.get_dummies(df, columns=['age_group'], drop_first=True)
    
    # Handle missing values
    df = df.fillna(df.median())
    
    # Separate features and target
    X = df.drop('task_completion_likelihood', axis=1)
    y = df['task_completion_likelihood']
    
    # Save column structure
    os.makedirs("models", exist_ok=True)
    joblib.dump(X.columns.tolist(), COLUMNS_PATH)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    
    return X_scaled, y

def prepare_data():
    """Main data processing workflow"""
    df = load_data()
    X, y = preprocess_data(df)
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()