import numpy as np
from tensorflow.python.keras.models import Sequential
from sklearn.base import BaseEstimator, ClassifierMixin

class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self._estimator_type = "classifier"
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, **kwargs):
        """Fit the model to the data."""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        """Predict class labels for samples in X."""
        proba = self.model.predict(X)
        return (proba > 0.5).astype(int)

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        # Convert input to numpy array and ensure correct shape
        X = np.array(X).astype('float32')
        
        # Get predictions
        predictions = self.model.predict(X, verbose=0)
        
        # For binary classification, return both class probabilities
        if predictions.shape[-1] == 1:  # Binary classification
            return np.hstack([1 - predictions, predictions])
        return predictions

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"model": self.model}

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    @property
    def classes(self):
        return self.classes_