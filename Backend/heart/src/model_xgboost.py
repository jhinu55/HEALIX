import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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

def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    predictions = model.predict(X_test)
    return predictions

def evaluate_model(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('../data/heart.csv')
    X, y = preprocess_data(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the XGBoost model
    model = train_xgboost(X_train, y_train)
    
    # Make predictions
    predictions = predict(model, X_test)
    
    # Evaluate the model
    accuracy, report = evaluate_model(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)

    from utils import save_model
    save_model(model, 'saved_models/xgboost_model.pkl')
    print("XGBoost model saved as 'saved_models/xgboost_model.pkl'")