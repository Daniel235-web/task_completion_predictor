# Description: This script contains the feature engineering code for the project.
import pandas as pd

def engineer_features(df):
    """Create new features from raw data"""
    # Productivity metric
    df['productivity'] = df['tasks_completed'] / (df['time_spent'] + 1e-6)
    
    # Age groupings
    bins = [0, 25, 35, 45, 55, 100]
    labels = ['18-25', '26-35', '36-45', '46-55', '56+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    
    return df