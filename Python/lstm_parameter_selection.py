# -*- coding: utf-8 -*-
"""
Created 2025/04/12
Converted to TensorFlow/Keras (CPU) 2025/11/03

@author: Mario
"""
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utility import get_station_name, load_and_prepare_data
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

###############################################################################
# === Parameters ===
LAG = 24
EPOCHS = 100
BATCH_SIZE = 64
LR = 0.001
HIDDEN_SIZE = 500
NUM_LAYERS = 3
DROPOUT = 0.2
PATIENCE = 5

###############################################################################
# === Configuration ===
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Feature Engineering ===
features = ['NO(ppm)', 'NO2(ppm)', 'U', 'V']
engineered_items = [target_item, 'NO(ppm)', 'NO2(ppm)', 'U', 'V']

# Rolling mean and std features
rolling_features = {}
for item in engineered_items:
    col_mean = f'{item}_roll_mean_3'
    col_std = f'{item}_roll_std_6'
    rolling_features[col_mean] = df[item].shift(1).rolling(3).mean()
    rolling_features[col_std] = df[item].shift(1).rolling(6).std()
    features += [col_mean, col_std]

df = pd.concat([df, pd.DataFrame(rolling_features, index=df.index)], axis=1)

# Difference features
diff_features = {}
diff_features[f'{target_item}_diff_1'] = df[target_item].shift(1).diff(1)
diff_features[f'{target_item}_diff_2'] = df[target_item].shift(1).diff(2)
diff_features[f'{target_item}_diff_3'] = df[target_item].shift(1).diff(3)
features += list(diff_features.keys())

for item in ['NO(ppm)', 'NO2(ppm)', 'U', 'V']:
    col_name = f'{item}_diff_3'
    diff_features[col_name] = df[item].shift(1).diff(3)
    features.append(col_name)

df = pd.concat([df, pd.DataFrame(diff_features, index=df.index)], axis=1)

# Time-related features
df["hour_sin"] = np.sin(2 * np.pi * df["時"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["時"] / 24)
df["dayofweek"] = df.index.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
features += ['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend']

target_col = [target_item]

# Drop missing rows
data_model = df.dropna(subset=features + target_col).copy()

###############################################################################
# === Train/Test Split (80/20) ===
n_samples = len(data_model)
train_size = int(0.8 * n_samples)

X_train_raw = data_model[features].iloc[:train_size]
y_train_raw = data_model[target_col].iloc[:train_size]
X_val_raw = data_model[features].iloc[train_size:]
y_val_raw = data_model[target_col].iloc[train_size:]

###############################################################################
# === Normalization (fit only on training data) ===
scaler_X = MinMaxScaler().fit(X_train_raw)
scaler_y = MinMaxScaler().fit(y_train_raw)

X_train = pd.DataFrame(scaler_X.transform(X_train_raw), columns=features)
X_val = pd.DataFrame(scaler_X.transform(X_val_raw), columns=features)
y_train = pd.DataFrame(scaler_y.transform(y_train_raw), columns=target_col)
y_val = pd.DataFrame(scaler_y.transform(y_val_raw), columns=target_col)

###############################################################################
# === Sequence Creation ===
def create_sequences(X: pd.DataFrame, y: pd.Series, lag: int):
    """Create sequences of length `lag` for LSTM input."""
    Xs, ys = [], []
    for i in range(len(X) - lag):
        Xs.append(X.iloc[i:i+lag].values)
        ys.append(y.iloc[i+lag])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, LAG)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, LAG)

print("X_train_seq shape:", X_train_seq.shape)
print("y_train_seq shape:", y_train_seq.shape)
print("X_val_seq shape:", X_val_seq.shape)
print("y_val_seq shape:", y_val_seq.shape)

###############################################################################
# === TensorFlow Dataset ===
train_ds = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train_seq)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val_seq, y_val_seq)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

###############################################################################
# === LSTM Model Definition ===
def build_lstm_model(input_shape, hidden_size, num_layers, dropout):
    """Build a multi-layer LSTM model."""
    model = Sequential()
    for i in range(num_layers):
        is_last = (i == num_layers - 1)
        if i == 0:
            model.add(LSTM(hidden_size,
                           return_sequences=not is_last,
                           dropout=dropout,
                           recurrent_dropout=0.0,
                           input_shape=input_shape))
        else:
            model.add(LSTM(hidden_size,
                           return_sequences=not is_last,
                           dropout=dropout,
                           recurrent_dropout=0.0))
    model.add(Dense(1))
    return model

input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
model = build_lstm_model(input_shape, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss='mse')
model.summary()

###############################################################################
# === Save Parameters ===
PARAMS_PATH = "lstm_best_params.json"
params = {
    "lag": LAG,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "dropout": DROPOUT,
    "patience": PATIENCE,
    "features": features
}
with open(PARAMS_PATH, "w", encoding="utf-8") as f:
    json.dump(params, f, indent=4, ensure_ascii=False)

###############################################################################
# === Callbacks (EarlyStopping + ModelCheckpoint) ===
MODEL_PATH = "lstm_best_model.keras"
checkpoint_cb = ModelCheckpoint(MODEL_PATH, monitor='val_loss', mode='min',
                                save_best_only=True, save_weights_only=False, verbose=1)
earlystop_cb = EarlyStopping(monitor='val_loss', patience=PATIENCE,
                             restore_best_weights=True, verbose=1)

###############################################################################
# === Training ===
history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=val_ds,
                    callbacks=[checkpoint_cb, earlystop_cb],
                    verbose=2)

train_losses = history.history.get('loss', [])
val_losses = history.history.get('val_loss', [])

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve (Keras)')
plt.legend()
plt.grid(True)
plt.show()

###############################################################################
# === Load Best Model ===
best_model = load_model(MODEL_PATH)

###############################################################################
# === Forecast on Validation Set ===
y_pred_scaled = best_model.predict(X_val_seq).flatten()
y_true_scaled = y_val_seq.flatten()

# Inverse transform only the target variable
y_pred_inv = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true_inv = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

###############################################################################
# === Evaluation Metrics ===
r2 = r2_score(y_true_inv, y_pred_inv)
mae = mean_absolute_error(y_true_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))

print(f"R² (original scale): {r2:.6f}")
print(f"MAE (original scale): {mae:.6f}")
print(f"RMSE (original scale): {rmse:.6f}")

with open("lstm_best_result.txt", "w", encoding="utf-8") as f:
    f.write(f"R² (original scale): {r2:.6f}\n")
    f.write(f"MAE (original scale): {mae:.6f}\n")
    f.write(f"RMSE (original scale): {rmse:.6f}\n")

###############################################################################
# === Plot Results ===
plt.figure(figsize=(12, 5))
plt.plot(y_true_inv[:300], label="True (ppm)")
plt.plot(y_pred_inv[:300], label="Predicted (ppm)")
plt.legend()
plt.title(f"{station_name} - LSTM Prediction vs True Values (ppm)")
plt.xlabel("Time step")
plt.ylabel("Ox (ppm)")
plt.grid(True)
plt.show()
