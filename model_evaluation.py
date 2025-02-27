# model_evaluation.py (updated)
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score,    recall_score, confusion_matrix)
from data_processing import prepare_data

MODEL_PATH = "models/task_predictor.joblib"

def evaluate_model():
    """Generate performance metrics and visualizations"""
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    X_train, X_test, y_train, y_test = prepare_data()
    model = joblib.load(MODEL_PATH)
    
    y_pred = model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    
    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('reports/confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    evaluate_model()