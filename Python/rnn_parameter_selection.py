# -*- coding: utf-8 -*-
"""
Created 2025/04/12
Converted to TensorFlow/Keras (CPU) 2025/11/03

RNN single-step forecast for Ox(ppm)
Author: Mario
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utility import get_station_name, load_and_prepare_data
from report import save_report_to_pdf
import os

###############################################################################
# === Parameters ===
LAG = 24
HORIZON = 1  # Forecast horizon (1 step ahead)
EPOCHS = 200
BATCH_SIZE = 128
LR = 0.0005
HIDDEN_SIZE = 80
NUM_LAYERS = 1
DROPOUT = 0
PATIENCE = 8

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

# Rolling mean and std
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

# Time-based features
df["hour_sin"] = np.sin(2 * np.pi * df["時"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["時"] / 24)
df["dayofweek"] = df.index.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
features += ['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend']

target_col = [target_item]
data_model = df.dropna(subset=features + target_col).copy()

###############################################################################
# === Train/Test Split ===
n_samples = len(data_model)
train_size = int(0.8 * n_samples)
X_train_raw = data_model[features].iloc[:train_size]
y_train_raw = data_model[target_col].iloc[:train_size]
X_val_raw = data_model[features].iloc[train_size:]
y_val_raw = data_model[target_col].iloc[train_size:]

###############################################################################
# === Normalization ===
scaler_X = MinMaxScaler().fit(X_train_raw)
scaler_y = MinMaxScaler().fit(y_train_raw)
X_train = pd.DataFrame(scaler_X.transform(X_train_raw), columns=features)
X_val = pd.DataFrame(scaler_X.transform(X_val_raw), columns=features)
y_train = pd.DataFrame(scaler_y.transform(y_train_raw), columns=target_col)
y_val = pd.DataFrame(scaler_y.transform(y_val_raw), columns=target_col)

###############################################################################
# === Sequence Creation ===
def create_sequences(X: pd.DataFrame, y: pd.Series, lag: int):
    """Create lagged sequences for RNN input."""
    Xs, ys = [], []
    for i in range(len(X) - lag):
        Xs.append(X.iloc[i:i+lag].values)
        ys.append(y.iloc[i+lag])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, LAG)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, LAG)

###############################################################################
# === TensorFlow Dataset ===
train_ds = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train_seq)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val_seq, y_val_seq)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

###############################################################################
# === RNN Model Definition ===
def build_rnn_model(input_shape, hidden_size, num_layers, dropout):
    """Build a multi-layer SimpleRNN model."""
    model = Sequential()
    for i in range(num_layers):
        is_last = (i == num_layers - 1)
        if i == 0:
            model.add(SimpleRNN(hidden_size,
                                return_sequences=not is_last,
                                dropout=dropout,
                                recurrent_dropout=0.0,
                                input_shape=input_shape))
        else:
            model.add(SimpleRNN(hidden_size,
                                return_sequences=not is_last,
                                dropout=dropout,
                                recurrent_dropout=0.0))
    model.add(Dense(1))
    return model

input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
model = build_rnn_model(input_shape, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss='mse')
model.summary()

###############################################################################
# === Save Parameters ===
PARAMS_PATH = "rnn_best_params.json"
params = {
    "lag": LAG,
    "horizon": HORIZON,
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

FEATURES_PATH = "rnn_features_used.json"
with open(FEATURES_PATH, "w", encoding="utf-8") as f:
    json.dump(features, f, indent=4, ensure_ascii=False)

###############################################################################
# === Callbacks ===
MODEL_PATH = "rnn_best_model.keras"
checkpoint_cb = ModelCheckpoint(MODEL_PATH, monitor='val_loss', mode='min',
                                save_best_only=True, verbose=1)
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

###############################################################################
# === Load Best Model ===
best_model = load_model(MODEL_PATH)

###############################################################################
# === Forecast ===
y_pred_scaled = best_model.predict(X_val_seq).flatten()
y_true_scaled = y_val_seq.flatten()
y_pred_inv = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true_inv = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

###############################################################################
# === Evaluation ===
r2 = r2_score(y_true_inv, y_pred_inv)
mae = mean_absolute_error(y_true_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))

with open("rnn_best_result.txt", "w", encoding="utf-8") as f:
    f.write(f"R² (original scale): {r2:.6f}\n")
    f.write(f"MAE (original scale): {mae:.6f}\n")
    f.write(f"RMSE (original scale): {rmse:.6f}\n")

###############################################################################
# === Visualization ===
figures = []

# Learning Curve
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss', color='gray')
ax1.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', color='lightgray', linestyle='dashed')
ax1.set_title(f'Learning Curve (RNN)\nBest Validation Loss: {min(val_losses):.5f}')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.legend()
ax1.grid(True)
figures.append(fig1)

# Predicted vs True
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(y_true_inv[-720:], label='True values', color='gray')
ax2.plot(y_pred_inv[-720:], label='RNN (with lag)', color='lightgray', linestyle='dashed')
ax2.set_title(f'RNN Regression with lag\nR²: {r2:.5f}')
ax2.set_xlabel('Samples')
ax2.set_ylabel(target_item)
ax2.legend()
ax2.grid(True)
figures.append(fig2)

# Residuals
residuals = y_true_inv - y_pred_inv
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.scatter(range(len(residuals)), residuals, alpha=0.5, s=10, color='gray')
ax3.axhline(0, color='black', linestyle='--', linewidth=1)
ax3.set_title('Residual Error Distribution')
ax3.set_xlabel('Samples')
ax3.set_ylabel('Residual (True - Predicted)')
ax3.grid(True)
figures.append(fig3)

# Histogram
res_mean = np.mean(residuals)
res_median = np.median(residuals)
hist_counts, bin_edges = np.histogram(residuals, bins=50)
mode_index = np.argmax(hist_counts)
res_mode = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2
fig4, ax4 = plt.subplots(figsize=(8, 4))
ax4.hist(residuals, bins=50, color='lightgray', edgecolor='gray', alpha=0.75)
ax4.axvline(res_mean, color='black', linestyle='--', linewidth=1, label=f'Mean = {res_mean:.5f}')
ax4.axvline(res_median, color='black', linestyle='-.', linewidth=1, label=f'Median = {res_median:.5f}')
ax4.axvline(res_mode, color='black', linestyle=':', linewidth=1, label=f'Mode = {res_mode:.5f}')
ax4.set_title('Histogram of Residuals')
ax4.set_xlabel('Residual value')
ax4.set_ylabel('Frequency')
ax4.legend()
ax4.grid(True)
figures.append(fig4)

plt.show()

###############################################################################
# === PDF Report ===
report_file = f'rnn_parameter_selection_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf'
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

params_info = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Number of data points in the train set": len(y_train),
    "Number of data points in the validation set": len(y_val),
    "Number of features used": len(features),
    "Model": "RNN",
    "Lag": LAG,
    "Epochs": EPOCHS,
    "Batch size": BATCH_SIZE,
    "Learning rate": LR,
    "Hidden size": HIDDEN_SIZE,
    "Number of layers": NUM_LAYERS,
    "Dropout": DROPOUT,
    "Patience": PATIENCE,
    "Best validation loss": float(f"{min(val_losses):.6f}")
}

params_info["Residuals mean"] = float(f"{res_mean:.6f}")
params_info["Residuals median"] = float(f"{res_median:.6f}")
params_info["Residuals mode"] = float(f"{res_mode:.6f}")
params_info["Residuals std"] = float(f"{np.std(residuals):.6f}")
params_info["Residuals skew"] = float(f"{pd.Series(residuals).skew():.3f}")
params_info["Residuals kurtosis"] = float(f"{pd.Series(residuals).kurtosis():.3f}")

errors = [(target_item, r2, mae, rmse)]

save_report_to_pdf(
    filename=report_path,
    title=f"RNN parameter selection Report - {station_name}",
    params=params_info,
    features=features,
    errors=errors,
    figures=figures
)

print(f"\n✅ Report saved as {report_path}")
