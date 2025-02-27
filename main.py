# Description: Main file to run the project
import argparse
from model_training import train_model
from model_evaluation import evaluate_model
from user_interaction import cli_interface, streamlit_interface
import os

def main():
    parser = argparse.ArgumentParser(description="Task Completion Predictor")
    parser.add_argument('--train', action='store_true', help="Retrain model")
    parser.add_argument('--web', action='store_true', help="Launch web UI")
    
    args = parser.parse_args()
    
    if args.train or not os.path.exists(MODEL_PATH):
        print("Training new model...")
        train_model()
        evaluate_model()
    
    if args.web:
        streamlit_interface()
    else:
        print("Use command line arguments to make predictions:")
        cli_interface()

if __name__ == "__main__":
    main()