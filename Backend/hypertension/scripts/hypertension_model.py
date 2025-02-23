#!/usr/bin/env python3
import json
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import joblib

# 1. First, add class weight handling and SMOTE for imbalanced data
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# 1. LOAD DATA
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'hypertension_data.csv')

def load_data(filepath):
    """
    Load CSV data into a pandas DataFrame.
    """
    df = pd.read_csv(filepath)
    return df

# 2. DATA PROCESSING & FEATURE ENGINEERING
def preprocess_data(df):
    """
    Clean and transform the raw data, and split features/target.
    We'll do minimal transformations, but you can add more as needed.
    """

    # Let's rename columns for convenience (optional)
    df.columns = [
        'Age', 'Sex', 'Education', 'Occupation', 'Monthly_Income', 'Residence',
        'Systolic_BP', 'Diastolic_BP', 'Elevated_Creatinine', 'Diabetes',
        'Family_History_CVD', 'Elevated_Cholesterol', 'Smoking', 'LVH', 'IHD',
        'CVD', 'Retinopathy', 'Treatment', 'Control_Status', 'Physical_Activity',
        'Dietary_Habits'
    ]

    # Example: We interpret "Control_Status" as the target
    # Map "Controlled" -> 1 and "Uncontrolled" -> 0 (or vice versa)
    df['Control_Status'] = df['Control_Status'].map({'Controlled': 1, 'Uncontrolled': 0})

    # Convert yes/no columns to 1/0
    # We'll assume any "Yes"/"No" type columns are as below:
    yes_no_cols = [
        'Elevated_Creatinine', 'Diabetes', 'Family_History_CVD',
        'Elevated_Cholesterol', 'Smoking', 'LVH', 'IHD', 'CVD',
        'Retinopathy', 'Physical_Activity'
    ]

    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # For "Dietary_Habits" we have "Unhealthy" or "Healthy" possibly
    # We'll map them as well
    df['Dietary_Habits'] = df['Dietary_Habits'].map({'Unhealthy': 0, 'Healthy': 1})

    # "Treatment" might have multiple categories e.g. "Combination Drugs", "Monotherapy", "NoTreatment" etc.
    # We'll do a simple label encoding or one-hot encoding for that:
    df['Treatment'].fillna('Unknown', inplace=True)  # handle missing
    df['Treatment'] = df['Treatment'].astype(str)

    # "Sex", "Education", "Occupation", "Monthly_Income", "Residence" are also categorical
    # We'll handle them via one-hot encoding or label encoding
    # We'll define them as categorical for pipeline
    cat_cols = ['Sex', 'Education', 'Occupation', 'Monthly_Income', 'Residence', 'Treatment']

    # Numeric columns
    numeric_cols = ['Age', 'Systolic_BP', 'Diastolic_BP']

    # Add interaction features
    df['BP_Product'] = df['Systolic_BP'] * df['Diastolic_BP']
    df['BP_Ratio'] = df['Systolic_BP'] / df['Diastolic_BP']
    df['Age_BP_Interaction'] = df['Age'] * df['Systolic_BP']
    
    # Add risk factors count
    risk_factors = ['Diabetes', 'Elevated_Cholesterol', 'Smoking', 
                   'Family_History_CVD', 'Physical_Activity']
    df['Risk_Factor_Count'] = df[risk_factors].sum(axis=1)
    
    # Update numeric_cols with new features
    numeric_cols.extend(['BP_Product', 'BP_Ratio', 'Age_BP_Interaction', 
                        'Risk_Factor_Count'])

    # Create X, y
    X = df[cat_cols + numeric_cols + yes_no_cols + ['Dietary_Habits']]
    y = df['Control_Status']

    return X, y, cat_cols, numeric_cols

def create_preprocessing_pipeline(cat_cols, numeric_cols):
    """
    Create a scikit-learn ColumnTransformer pipeline that handles
    categorical columns via OneHotEncoder and numeric columns via StandardScaler.
    """
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer

    # For numeric columns, we can do scaling
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # For categorical columns, we do one-hot
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    return preprocessor

def evaluate_predictions(y_true, y_pred, y_proba):
    """
    Evaluate model predictions with probability scores
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)

    print(f"Model Performance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    # Print probability distribution
    print("\nProbability Distribution Analysis:")
    proba_percentiles = np.percentile(y_proba, [25, 50, 75])
    print(f"25th percentile: {proba_percentiles[0]:.2%}")
    print(f"Median: {proba_percentiles[1]:.2%}")
    print(f"75th percentile: {proba_percentiles[2]:.2%}")

    return accuracy, precision, recall, f1, auc

def analyze_feature_importance(model, feature_names):
    importance = model.named_steps['classifier'].feature_importances_
    feature_imp = pd.DataFrame({'feature': feature_names, 
                              'importance': importance})
    feature_imp = feature_imp.sort_values('importance', 
                                         ascending=False).head(10)
    print("\nTop 10 Important Features:")
    print(feature_imp)
    return feature_imp

from sklearn.model_selection import cross_val_score

def perform_cross_validation(model, X, y, cv=5):
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    print("\nCross-validation scores:", cv_scores)
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 3. MAIN FUNCTION
def main():
    # Load data
    df = load_data(DATA_PATH)
    print(f"Data shape: {df.shape}")

    # Preprocess
    X, y, cat_cols, numeric_cols = preprocess_data(df)
    print("Preprocessing done. X shape:", X.shape, "y shape:", y.shape)

    # Create pipeline
    preprocessor = create_preprocessing_pipeline(cat_cols, numeric_cols)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Apply preprocessing before SMOTE
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Handle class imbalance with preprocessed data
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_preprocessed, y_train)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y), class_weights))

    # 4. Modify XGBoost parameters for probability calibration
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric=['auc', 'logloss'],
        random_state=42,
        scale_pos_weight=class_weights[1]/class_weights[0],
        max_delta_step=1,
        enable_categorical=True,
        tree_method='hist',  # Faster training
        early_stopping_rounds=10  # Prevent overfitting
    )

    # Updated parameter grid with proper prefixes
    param_grid = {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__max_depth': [3, 4, 5, 6, 8],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__subsample': [0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.8, 0.9, 1.0],
        'classifier__gamma': [0, 0.1, 0.2],
        'classifier__min_child_weight': [1, 3, 5],
        'classifier__reg_alpha': [0, 0.1, 0.5],
        'classifier__reg_lambda': [0.1, 1.0, 5.0]
    }

    # Create pipeline with XGBoost
    xgb_pipeline = Pipeline(steps=[
        ('classifier', xgb_clf)
    ])

    # Updated GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=param_grid,
        scoring=['f1', 'precision', 'recall', 'roc_auc'],
        refit='f1',  # Choose primary metric
        cv=5,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    print("Starting Grid Search for XGBoost...")
    # Suppress warnings during fit
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        grid_search.fit(X_train_balanced, y_train_balanced)
    print("Best params:", grid_search.best_params_)

    best_xgb_model = grid_search.best_estimator_

    # Evaluate XGBoost with probabilities
    y_pred = best_xgb_model.predict(X_test_preprocessed)
    y_proba = best_xgb_model.predict_proba(X_test_preprocessed)[:,1]

    print("\nXGBoost Model Results:")
    xgb_metrics = evaluate_predictions(y_test, y_pred, y_proba)

    # Sample probability interpretations
    print("\nExample Probability Interpretations:")
    for prob in [0.2, 0.5, 0.8]:
        risk_level = "Low" if prob < 0.4 else "Moderate" if prob < 0.7 else "High"
        print(f"Probability {prob:.1%} - {risk_level} risk of hypertension")

    # Save model with probability calibration
    MODEL_PATH = os.path.join(BASE_DIR, 'scripts', 'final_xgb_model.joblib')
    joblib.dump(best_xgb_model, MODEL_PATH)
    print(f"\nSaved probability-calibrated model to {MODEL_PATH}")

    # Function to predict probability for new data
    def predict_hypertension_risk(model, new_data, preprocessor):
        """
        Predict hypertension risk probability for new data
        """
        processed_data = preprocessor.transform(new_data)
        probability = model.predict_proba(processed_data)[:,1]
        return probability

    # 5. Create ensemble with proper pipelines for each model
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )

    lr_clf = LogisticRegression(
        random_state=42,
        max_iter=2000,
        class_weight='balanced',
        C=0.1
    )

    # Use weights for ensemble members
    ensemble_clf = VotingClassifier(
        estimators=[
            ('xgb', grid_search.best_estimator_),
            ('rf', rf_clf),
            ('lr', lr_clf)
        ],
        voting='soft',

        weights=[2, 1, 1]
    )

    # Train ensemble on balanced data
    print("Training ensemble model...")
    ensemble_clf.fit(X_train_balanced, y_train_balanced)
    y_ens_pred = ensemble_clf.predict(X_test_preprocessed)
    y_ens_proba = ensemble_clf.predict_proba(X_test_preprocessed)[:,1]

    ensemble_metrics = evaluate_predictions(y_test, y_ens_pred, y_ens_proba)

    # 6. Save final model (XGBoost or ensemble) to disk
    MODEL_PATH = os.path.join(BASE_DIR, 'scripts', 'final_xgb_model.joblib')
    joblib.dump(best_xgb_model, MODEL_PATH)
    print(f"Saved XGBoost model to {MODEL_PATH}")

    # If you prefer saving ensemble model, do:
    # joblib.dump(ensemble_clf, 'ensemble_model.joblib')

    # After grid search
    print("\nPerforming cross-validation...")
    perform_cross_validation(best_xgb_model, X_train_balanced, y_train_balanced)
    
    # Analyze feature importance
    feature_names = (numeric_cols + 
                    [f"{col}_{val}" for col in cat_cols 
                     for val in preprocessor.named_transformers_['cat']
                     .named_steps['onehot'].get_feature_names_out([col])])
    analyze_feature_importance(best_xgb_model, feature_names)

    # Create models directory if it doesn't exist
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save both preprocessor and model
    print("\nPerforming cross-validation...")
    perform_cross_validation(best_xgb_model, X_train_balanced, y_train_balanced)
    
    # Analyze feature importance
    feature_names = (numeric_cols + 
                    [f"{col}_{val}" for col in cat_cols 
                     for val in preprocessor.named_transformers_['cat']
                     .named_steps['onehot'].get_feature_names_out([col])])
    analyze_feature_importance(best_xgb_model, feature_names)
    MODEL_PATH = os.path.join(MODELS_DIR, 'hypertension_model.joblib')
    PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    
    print("\nSaving models...")
    # Save the best model (either XGBoost or ensemble)
    if xgb_metrics[3] > ensemble_metrics[3]:  # Compare F1 scores
        print("Saving XGBoost model (better performance)...")
        model_to_save = best_xgb_model
    else:
        print("Saving ensemble model (better performance)...")
        model_to_save = ensemble_clf

    # Save both model and preprocessor
    joblib.dump(model_to_save, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved preprocessor to: {PREPROCESSOR_PATH}")

    # Save model metadata
    metadata = {
        'model_type': type(model_to_save).__name__,
        'feature_columns': cat_cols + numeric_cols,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'performance_metrics': {
            'accuracy': float(xgb_metrics[0]),
            'precision': float(xgb_metrics[1]),
            'recall': float(xgb_metrics[2]),
            'f1_score': float(xgb_metrics[3]),
            'auc': float(xgb_metrics[4])
        }
    }
    
    METADATA_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Saved model metadata to: {METADATA_PATH}")

if __name__ == "__main__":
    main()
