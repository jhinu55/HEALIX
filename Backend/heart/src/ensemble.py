import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer as KInputLayer
from utils import load_model as load_pkl_model
from utils import preprocess_data
from model_wrapper import KerasClassifierWrapper

def custom_input_layer(*args, **kwargs):
    # Remove the unknown 'batch_shape' argument, if present
    kwargs.pop('batch_shape', None)
    
    # Ensure shape is provided either via 'shape', 'input_shape', or a global default
    if not args and 'shape' not in kwargs:
        if 'input_shape' in kwargs:
            kwargs['shape'] = kwargs['input_shape']
        elif 'DEFAULT_INPUT_SHAPE' in globals():
            kwargs['shape'] = globals()['DEFAULT_INPUT_SHAPE']
        else:
            raise ValueError("Input shape must be provided")
    
    return KInputLayer(*args, **kwargs)

def load_data(filepath):
    data = pd.read_csv(filepath)
    data = preprocess_data(data) 
    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']
    return X, y

def ensemble_predict_manual(X_test):
    try:
        # Set a global default input shape
        global DEFAULT_INPUT_SHAPE
        DEFAULT_INPUT_SHAPE = (X_test.shape[1],)

        # Load models with proper input shape handling
        ann_model = load_model(
            'saved_models/ann_model.h5',
            custom_objects={'InputLayer': custom_input_layer}
        )
        
        # Explicitly build the model (if necessary)
        ann_model.build((None, X_test.shape[1]))
        
        ann_wrapper = KerasClassifierWrapper(ann_model)
        xgb_model = load_pkl_model('saved_models/xgboost_model.pkl')
        rf_model = load_pkl_model('saved_models/randomforest_model.pkl')
        
        # Prepare data
        X_test_np = X_test.to_numpy().astype('float32')
        
        # Get predictions
        ann_proba = ann_wrapper.predict_proba(X_test_np)
        xgb_proba = xgb_model.predict_proba(X_test)
        rf_proba = rf_model.predict_proba(X_test)
        
        # Ensemble predictions (soft voting)
        avg_proba = (ann_proba + xgb_proba + rf_proba) / 3.0
        predictions = (avg_proba[:,1] > 0.5).astype(int)
        
        return predictions
        
    except Exception as e:
        print(f"Error in ensemble prediction: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    X, y = load_data('../data/heart.csv')
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # When saving the ANN model in model_ann.py
    
    predictions = ensemble_predict_manual(X_test)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, predictions)
    print(f"Ensemble Accuracy (manual soft voting): {accuracy * 100:.2f}%")