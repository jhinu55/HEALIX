import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load diabetes dataset"""
        try:
            # Load the diabetes dataset
            # Expected columns: Pregnancies, Glucose, BloodPressure, SkinThickness, 
            # Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
            data = pd.read_csv(filepath)
            logging.info(f"Loaded dataset with shape: {data.shape}")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return None

    def preprocess_data(self, data):
        """Preprocess the data"""
        try:
            # Handle missing values
            data = data.replace(0, np.nan)
            data = data.fillna(data.mean())
            
            # Split features and target
            X = data.drop('Outcome', axis=1)
            y = data['Outcome']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            return None, None

    def train_model(self, X, y):
        """Train the Random Forest model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Initialize and train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            logging.info("\nModel Performance:")
            logging.info("\nConfusion Matrix:")
            logging.info(confusion_matrix(y_test, y_pred))
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred))
            
            return X_test, y_test
        except Exception as e:
            logging.error(f"Error training model: {e}")
            return None, None

    def save_model(self, model_path='diabetes_model.joblib', scaler_path='scaler.joblib'):
        """Save the trained model and scaler"""
        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            logging.info(f"Model saved to {model_path}")
            logging.info(f"Scaler saved to {scaler_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def predict(self, features):
        """Make predictions using the trained model"""
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            # Make prediction
            prediction = self.model.predict(features_scaled)
            probability = self.model.predict_proba(features_scaled)
            return prediction[0], probability[0]
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            return None, None

def main():
    # Initialize predictor
    predictor = DiabetesPredictor()
    
    # Load and preprocess data
    data = predictor.load_data('diabetes_dataset.csv')
    if data is not None:
        X_scaled, y = predictor.preprocess_data(data)
        if X_scaled is not None:
            # Train model
            X_test, y_test = predictor.train_model(X_scaled, y)
            if X_test is not None:
                # Save model
                predictor.save_model()
                
                # Test prediction
                sample = X_test[0:1]
                prediction, probability = predictor.predict(sample)
                if prediction is not None:
                    logging.info(f"\nSample prediction: {prediction}")
                    logging.info(f"Probability: {probability}")

if __name__ == "__main__":
    main()