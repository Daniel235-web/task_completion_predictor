# Description: Train and save a classifier for task prediction
from sklearn.ensemble import RandomForestClassifier
import joblib
from data_processing import prepare_data

MODEL_PATH = "models/task_predictor.joblib"

def train_model():
    """Train and save classifier"""
    X_train, X_test, y_train, y_test = prepare_data()
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    joblib.dump(model, MODEL_PATH)
    return model

if __name__ == "__main__":
    train_model()