import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import save_model

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Convert categorical variables to numerical
    data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})
    data['ChestPainType'] = data['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
    data['RestingECG'] = data['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
    data['ExerciseAngina'] = data['ExerciseAngina'].map({'N': 0, 'Y': 1})
    data['ST_Slope'] = data['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})
    
    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']
    return X, y

def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    return rf_model

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('../data/heart.csv')
    X, y = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    rf_model = train_random_forest(X_train, y_train)
    
    # Evaluate model
    predictions = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Random Forest Accuracy: {accuracy * 100:.2f}%')
    
    # Save model
    save_model(rf_model, 'saved_models/randomforest_model.pkl')
    print("Random Forest model saved as 'saved_models/randomforest_model.pkl'")