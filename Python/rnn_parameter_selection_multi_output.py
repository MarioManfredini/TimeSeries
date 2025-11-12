# -*- coding: utf-8 -*-
"""
Created 2025/11/08
Author: Mario
Description:
Multi-step (24-hour horizon) forecasting of Ox(ppm) using SimpleRNN.
"""
import os
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

###############################################################################
# === Parameters ===
LAG = 24
HORIZON = 24   # Forecast horizon
EPOCHS = 200
BATCH_SIZE = 128
LR = 0.0005
HIDDEN_SIZE = 20
NUM_LAYERS = 1
DROPOUT = 0.0
PATIENCE = 8

###############################################################################
# === Configuration ===
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'
MODEL_PATH = "rnn_best_model_multi_output.keras"
PARAMS_PATH = "rnn_best_params_multi_output.json"
FEATURES_PATH = "rnn_features_used_multi_output.json"

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

###############################################################################
# === Create Multi-step Targets ===
# Generate future columns: Ox_t+1, Ox_t+2, ..., Ox_t+HORIZON
for i in range(1, HORIZON + 1):
    df[f"{target_item}_t+{i:02d}"] = df[target_item].shift(-i)

target_cols = [f"{target_item}_t+{i:02d}" for i in range(1, HORIZON + 1)]

# Drop missing rows (at the end due to shifting)
data_model = df.dropna(subset=features + target_cols).copy()

###############################################################################
# === Train/Test Split (80/20) ===
n_samples = len(data_model)
train_size = int(0.8 * n_samples)

X_train_raw = data_model[features].iloc[:train_size]
y_train_raw = data_model[target_cols].iloc[:train_size]
X_val_raw = data_model[features].iloc[train_size:]
y_val_raw = data_model[target_cols].iloc[train_size:]

###############################################################################
# === Normalization (fit only on training data) ===
scaler_X = MinMaxScaler().fit(X_train_raw)
scaler_y = MinMaxScaler().fit(y_train_raw)

X_train = pd.DataFrame(scaler_X.transform(X_train_raw), columns=features)
X_val = pd.DataFrame(scaler_X.transform(X_val_raw), columns=features)
y_train = pd.DataFrame(scaler_y.transform(y_train_raw), columns=target_cols)
y_val = pd.DataFrame(scaler_y.transform(y_val_raw), columns=target_cols)

###############################################################################
# === Sequence Creation ===
def create_sequences(X: pd.DataFrame, y: pd.DataFrame, lag: int):
    """Create sequences of length `lag` for RNN input and multi-step output."""
    Xs, ys = [], []
    for i in range(len(X) - lag):
        Xs.append(X.iloc[i:i+lag].values)
        ys.append(y.iloc[i+lag].values)  # Multi-step targets
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
# === RNN Model Definition ===
def build_rnn_model(input_shape, hidden_size, num_layers, dropout, output_size):
    """Build a SimpleRNN model for multi-step forecasting."""
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
    model.add(Dense(output_size))
    return model

input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
model = build_rnn_model(input_shape, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, HORIZON)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss='mse')
model.summary()

###############################################################################
# === Save Parameters ===
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
print(f"✅ Saved parameters to {PARAMS_PATH}")

with open(FEATURES_PATH, "w", encoding="utf-8") as f:
    json.dump(features, f, indent=4, ensure_ascii=False)
print(f"✅ Saved features list to {FEATURES_PATH}")

###############################################################################
# === Callbacks ===
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

###############################################################################
# === Load Best Model ===
best_model = load_model(MODEL_PATH)

###############################################################################
# === Forecast on Validation Set ===
y_pred_scaled = best_model.predict(X_val_seq)
y_true_scaled = y_val_seq

# Inverse transform all target steps
y_pred_inv = scaler_y.inverse_transform(y_pred_scaled)
y_true_inv = scaler_y.inverse_transform(y_true_scaled)

# Evaluate only the first horizon step (t+1) for comparability
r2 = r2_score(y_true_inv[:, 0], y_pred_inv[:, 0])
mae = mean_absolute_error(y_true_inv[:, 0], y_pred_inv[:, 0])
rmse = np.sqrt(mean_squared_error(y_true_inv[:, 0], y_pred_inv[:, 0]))

print(f"R² (t+1): {r2:.6f}")
print(f"MAE (t+1): {mae:.6f}")
print(f"RMSE (t+1): {rmse:.6f}")

with open("rnn_best_result_multi_output.txt", "w", encoding="utf-8") as f:
    f.write(f"R² (t+1): {r2:.6f}\n")
    f.write(f"MAE (t+1): {mae:.6f}\n")
    f.write(f"RMSE (t+1): {rmse:.6f}\n")

###############################################################################
# === Report Generation ===
# Residuals and figures are based only on the first horizon (t+1)

# === Save Figures ===
figures = []

# Learning Curve
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='gray')
ax1.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='lightgray', linestyle='dashed')
ax1.set_title(f'Learning Curve (SimpleRNN)\nBest Validation Loss: {min(val_losses):.5f}')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.legend()
ax1.grid(True)
fig1.tight_layout()
figures.append(fig1)

# Predicted vs Real
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(y_true_inv[-720:], label='True values', color='gray')
ax2.plot(y_pred_inv[-720:], label='RNN (with lag)', color='lightgray', linestyle='dashed')
ax2.set_title(f'SimpleRNN Regression\nR²: {r2:.5f}')
ax2.set_xlabel('Samples')
ax2.set_ylabel(target_item)
ax2.legend()
ax2.grid(True)
fig2.tight_layout()
figures.append(fig2)

# Residual Errors
residuals = y_true_inv.flatten() - y_pred_inv.flatten()
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.scatter(range(len(residuals)), residuals, alpha=0.5, s=10, color='gray')
ax3.axhline(0, color='black', linestyle='--', linewidth=1)
ax3.set_xlabel('Samples')
ax3.set_ylabel('Error (True - Predicted)')
ax3.set_title('Distribution of Residual Errors')
ax3.grid(True)
fig3.tight_layout()
figures.append(fig3)

# Histogram of Residuals
res_mean = np.mean(residuals)
res_median = np.median(residuals)
# Estimate mode using histogram (most frequent bin center)
hist_counts, bin_edges = np.histogram(residuals, bins=50)
mode_index = np.argmax(hist_counts)
res_mode = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2

fig4, ax4 = plt.subplots(figsize=(8, 4))
ax4.hist(residuals, bins=50, color='lightgray', edgecolor='gray', alpha=0.75)
ax4.axvline(res_mean, color='black', linestyle='--', linewidth=1, label=f'Mean = {res_mean:.5f}')
ax4.axvline(res_median, color='black', linestyle='-.', linewidth=1, label=f'Median = {res_median:.5f}')
ax4.axvline(res_mode, color='black', linestyle=':', linewidth=1, label=f'Mode = {res_mode:.5f}')
ax4.set_title('Histogram of Residuals – Distribution & Central Tendency')
ax4.set_xlabel('Residual value')
ax4.set_ylabel('Frequency')
ax4.legend()
ax4.grid(True)
fig4.tight_layout()
figures.append(fig4)

plt.show()

###############################################################################
# === Save Report to PDF ===
report_file = f'rnn_parameter_selection_multi_output_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf'
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

params_info = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Number of data points in the train set": len(y_train),
    "Number of data points in the validation set": len(y_val),
    "Number of features": len(features),
    "Model": "SimpleRNN",
    "Lag (sequence length)": LAG,
    "Epochs": EPOCHS,
    "Batch size": BATCH_SIZE,
    "Learning rate": LR,
    "Horizon": HORIZON,
    "Hidden size": HIDDEN_SIZE,
    "Layers": NUM_LAYERS,
    "Dropout": DROPOUT,
    "Patience": PATIENCE,
    "Best val loss": float(f"{min(val_losses):.6f}"),
    "R² (t+1)": float(f"{r2:.6f}"),
    "MAE (t+1)": float(f"{mae:.6f}"),
    "RMSE (t+1)": float(f"{rmse:.6f}"),
    "Residual mean": float(f"{res_mean:.6f}"),
    "Residual median": float(f"{res_median:.6f}"),
    "Residual mode": float(f"{res_mode:.6f}")
}

# --- Residuals and Predictions statistics ---
params_info["Predictions mean"] = float(f"{np.mean(y_pred_inv):.3f}")
params_info["Predictions std"] = float(f"{np.std(y_pred_inv):.3f}")
params_info["True mean"] = float(f"{np.mean(y_true_inv):.3f}")
params_info["True std"] = float(f"{np.std(y_true_inv):.3f}")
params_info["Residuals std"] = float(f"{np.std(residuals):.6f}")
params_info["Residuals skew"] = float(f"{pd.Series(residuals).skew():.3f}")
params_info["Residuals kurtosis"] = float(f"{pd.Series(residuals).kurtosis():.3f}")

errors = [
    (target_item, r2, mae, rmse)
]

save_report_to_pdf(
    filename=report_path,
    title=f"RNN parameter selection Report - {station_name}",
    params=params_info,
    features=features,
    errors=errors,
    figures=figures
)

print(f"\n✅ Report saved as {report_path}")
