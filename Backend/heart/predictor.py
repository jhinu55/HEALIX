import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model as keras_load_model
from src.utils import load_model, preprocess_data
from src.model_wrapper import KerasClassifierWrapper

def process_input(user_input):
    """
    Processes a comma-separated string into a 1-row DataFrame.
    Converts natural language entries (e.g., "male", "ata") into the expected format.
    Expected order:
      Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
      RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
    """
    fields = user_input.split(',')
    if len(fields) != 11:
        print("Error: Expected 11 values, but got", len(fields))
        return None

    try:
        data = {}
        # Age
        data["Age"] = float(fields[0].strip())
        
        # Sex: convert 'male'/'female' to 'M'/'F'
        sex = fields[1].strip().lower()
        if sex in ['male', 'm']:
            data["Sex"] = "M"
        elif sex in ['female', 'f']:
            data["Sex"] = "F"
        else:
            data["Sex"] = fields[1].strip()

        # ChestPainType: assume the user enters one of ATA, NAP, ASY, TA (case-insensitive)
        data["ChestPainType"] = fields[2].strip().upper()

        # RestingBP
        data["RestingBP"] = float(fields[3].strip())

        # Cholesterol
        data["Cholesterol"] = float(fields[4].strip())

        # FastingBS (assumed numeric, e.g., 0 or 1)
        data["FastingBS"] = float(fields[5].strip())

        # RestingECG: convert common names to expected values
        re_input = fields[6].strip().lower()
        if re_input == "normal":
            data["RestingECG"] = "Normal"
        elif re_input in ["st", "stt"]:
            data["RestingECG"] = "ST"
        elif re_input == "lvh":
            data["RestingECG"] = "LVH"
        else:
            data["RestingECG"] = fields[6].strip()

        # MaxHR
        data["MaxHR"] = float(fields[7].strip())

        # ExerciseAngina: convert 'yes'/'no' to 'Y'/'N'
        ea = fields[8].strip().lower()
        if ea in ['yes', 'y']:
            data["ExerciseAngina"] = "Y"
        elif ea in ['no', 'n']:
            data["ExerciseAngina"] = "N"
        else:
            data["ExerciseAngina"] = fields[8].strip()

        # Oldpeak
        data["Oldpeak"] = float(fields[9].strip())

        # ST_Slope: capitalize first letter (e.g., Up, Flat, Down)
        data["ST_Slope"] = fields[10].strip().capitalize()

        # Create DataFrame
        df = pd.DataFrame([data])
        return df

    except Exception as ex:
        print("Error converting input:", ex)
        return None

def main():
    # Define expected columns in order
    columns = [
        "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", 
        "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
    ]

    print("Enter comma-separated values for:")
    print(", ".join(columns))
    print("Example: 40, male, ata, 140, 289, 0, normal, 172, no, 0, up")

    user_input = input("\nEnter values: ")
    df = process_input(user_input)
    if df is None:
        print("Input processing failed.")
        return

    # Optionally, further process the DataFrame to map categorical values to numeric.
    df = preprocess_data(df)

    # Load individual models
    ann_model = keras_load_model('src/saved_models/ann_model.h5')
    ann_wrapper = KerasClassifierWrapper(ann_model)

    xgb_model = load_model('src/saved_models/xgboost_model.pkl')
    rf_model = load_model('src/saved_models/randomforest_model.pkl')

    # Get probability predictions from each model. 
    # Use df.to_numpy() for the Keras model while others can use the DataFrame.
    ann_proba = ann_wrapper.predict_proba(df.to_numpy())
    xgb_proba = xgb_model.predict_proba(df)
    rf_proba = rf_model.predict_proba(df)

    # Average probabilities (soft voting)
    avg_proba = (ann_proba + xgb_proba + rf_proba) / 3.0
    probability = avg_proba[0, 1]  # probability of heart disease

    # Final prediction threshold (0.5 by default)
    prediction = 1 if probability > 0.5 else 0

    print("\nFinal Ensemble Prediction:", "Heart Disease" if prediction == 1 else "No Heart Disease")
    print("Average Probability of Heart Disease:", probability,"%")

if __name__ == '__main__':
    main()
