import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import joblib

# Load datasets
train_file = r'E:\backend\archive\01-12\working\export_dataframe_proc.csv'
test_file = r'E:\backend\archive\01-12\working\export_tests_proc.csv'

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Clean column names
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# Select features and target
features = ["Source Port", "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets"]
target = "Label"

# Check dataset balance
print("Original Dataset Class Distribution:\n", train_df[target].value_counts())

# Apply SMOTE for balancing
X_train, y_train = train_df[features], train_df[target]
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Normalize features
scaler = MinMaxScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
test_df[features] = scaler.transform(test_df[features])  # Use transform (not fit_transform)
joblib.dump(scaler, "scaler.pkl")

# Prepare data for LSTM
X_train = np.array(X_train_resampled).reshape(-1, 1, len(features))
y_train = np.array(y_train_resampled)
X_test = np.array(test_df[features]).reshape(-1, 1, len(features))
y_test = np.array(test_df[target])

# Build Improved LSTM Model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, input_shape=(1, len(features)))),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# Train model with validation data
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Save model
model.save("ddos_lstm_model.h5")
print("Model training complete and saved as ddos_lstm_model.h5")

# Plot accuracy and loss
plt.figure(figsize=(10, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Model Predictions and Evaluation
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

print("Classification Report:\n", classification_report(y_test, y_pred_binary))

