import tensorflow_config
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from tensorflow.python.distribute import input_lib
if not hasattr(input_lib, 'DistributedDatasetInterface'):
    input_lib.DistributedDatasetInterface = input_lib.DistributedDatasetSpec

# Load the dataset
data = pd.read_csv('../data/heart.csv')

# Preprocess the data
data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})
data['ChestPainType'] = data['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
data['RestingECG'] = data['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
data['ExerciseAngina'] = data['ExerciseAngina'].map({'N': 0, 'Y': 1})
data['ST_Slope'] = data['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})

# Split features and target
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to numpy arrays instead of TensorFlow tensors
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# Convert data to TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Batch the training data
BATCH_SIZE = 32
train_dataset = train_dataset.shuffle(len(X_train)).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

def create_ann_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train model
ann_model = create_ann_model((X_train.shape[1],))
history = ann_model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    verbose=1
)

# Evaluate model
loss, accuracy = ann_model.evaluate(test_dataset)
print(f'Test Accuracy: {accuracy:.4f}')

# Create directory if it doesn't exist
import os
os.makedirs('saved_models', exist_ok=True)

# Save the model
ann_model.build((None, X_train.shape[1]))  # Build with explicit input shape
ann_model.save('saved_models/ann_model.h5', save_format='h5')
print("ANN model saved as 'saved_models/ann_model.h5'")