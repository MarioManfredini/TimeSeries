# -*- coding: utf-8 -*-
"""
Created 2025/11/03
Author: Mario
Description:
Multi-step (24-hour horizon) forecasting of Ox(ppm) using MLP (Multi-Layer Perceptron).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
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
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.1
PATIENCE = 10

###############################################################################
# === Configuration ===
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'

MODEL_PATH = "mlp_best_model_multi_output.keras"
PARAMS_PATH = "mlp_best_params_multi_output.json"
FEATURES_PATH = "mlp_features_used_multi_output.json"

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Feature Engineering ===
features = ['NO(ppm)', 'NO2(ppm)', 'U', 'V']
engineered_items = [target_item, 'NO(ppm)', 'NO2(ppm)', 'U', 'V']

# Rolling mean and std features
for item in engineered_items:
    df[f'{item}_roll_mean_3'] = df[item].shift(1).rolling(3).mean()
    df[f'{item}_roll_std_6'] = df[item].shift(1).rolling(6).std()
    features += [f'{item}_roll_mean_3', f'{item}_roll_std_6']

# Difference features
for item in engineered_items:
    df[f'{item}_diff_1'] = df[item].shift(1).diff(1)
    df[f'{item}_diff_3'] = df[item].shift(1).diff(3)
    features += [f'{item}_diff_1', f'{item}_diff_3']

# Time-related features
df["hour_sin"] = np.sin(2 * np.pi * df["時"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["時"] / 24)
df["dayofweek"] = df.index.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
features += ['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend']

###############################################################################
# === Multi-step Targets ===
for i in range(1, HORIZON + 1):
    df[f"{target_item}_t+{i:02d}"] = df[target_item].shift(-i)
target_cols = [f"{target_item}_t+{i:02d}" for i in range(1, HORIZON + 1)]

df = df.dropna(subset=features + target_cols)

###############################################################################
# === Create Supervised Data for MLP ===
def create_supervised_data(df, features, target_cols, lag):
    """Create lagged features for MLP."""
    X, y = [], []
    for i in range(lag, len(df)):
        past = df[features].iloc[i - lag:i].values.flatten()
        X.append(past)
        y.append(df[target_cols].iloc[i].values)
    X = np.array(X)
    y = np.array(y)
    return X, y

X, y = create_supervised_data(df, features, target_cols, LAG)

###############################################################################
# === Train/Test Split ===
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:], y[train_size:]

###############################################################################
# === Normalization ===
scaler_X = MinMaxScaler().fit(X_train)
scaler_y = MinMaxScaler().fit(y_train)
X_train = scaler_X.transform(X_train)
X_val = scaler_X.transform(X_val)
y_train = scaler_y.transform(y_train)
y_val = scaler_y.transform(y_val)

###############################################################################
# === Build MLP Model ===
def build_mlp_model(input_dim, output_dim, hidden_size, num_layers, dropout):
    model = Sequential()
    model.add(Dense(hidden_size, activation='relu', input_dim=input_dim))
    for _ in range(num_layers - 1):
        model.add(Dense(hidden_size, activation='relu'))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(output_dim))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss='mse')
    return model

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = build_mlp_model(input_dim, output_dim, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
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
                                save_best_only=True, verbose=1)
earlystop_cb = EarlyStopping(monitor='val_loss', patience=PATIENCE,
                             restore_best_weights=True, verbose=1)

###############################################################################
# === Train ===
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[checkpoint_cb, earlystop_cb],
                    verbose=2)

###############################################################################
# === Load Best Model ===
best_model = load_model(MODEL_PATH)

###############################################################################
# === Evaluate ===
y_pred = best_model.predict(X_val)
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_true_inv = scaler_y.inverse_transform(y_val)

r2 = r2_score(y_true_inv[:, 0], y_pred_inv[:, 0])
mae = mean_absolute_error(y_true_inv[:, 0], y_pred_inv[:, 0])
rmse = np.sqrt(mean_squared_error(y_true_inv[:, 0], y_pred_inv[:, 0]))

print(f"R² (t+1): {r2:.6f}")
print(f"MAE (t+1): {mae:.6f}")
print(f"RMSE (t+1): {rmse:.6f}")

###############################################################################
# === Visualization ===
figures = []

# Learning curve
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(history.history['loss'], label='Train Loss', color='gray')
ax1.plot(history.history['val_loss'], label='Val Loss', color='lightgray', linestyle='dashed')
ax1.set_title(f"MLP Learning Curve – Best val_loss: {min(history.history['val_loss']):.5f}")
ax1.legend()
ax1.grid(True)
figures.append(fig1)

# Predicted vs True
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(y_true_inv[-720:, 0], color='gray', label='True')
ax2.plot(y_pred_inv[-720:, 0], color='lightgray', linestyle='dashed', label='MLP')
ax2.set_title(f"MLP Forecast (t+1)\nR²: {r2:.5f}")
ax2.legend()
ax2.grid(True)
figures.append(fig2)

# Residuals
residuals = y_true_inv[:, 0] - y_pred_inv[:, 0]
res_mean = np.mean(residuals)
res_median = np.median(residuals)
hist_counts, bin_edges = np.histogram(residuals, bins=50)
mode_index = np.argmax(hist_counts)
res_mode = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.hist(residuals, bins=50, color='lightgray', edgecolor='gray', alpha=0.75)
ax3.axvline(res_mean, color='black', linestyle='--', label=f"Mean={res_mean:.4f}")
ax3.axvline(res_median, color='black', linestyle='-.', label=f"Median={res_median:.4f}")
ax3.axvline(res_mode, color='black', linestyle=':', label=f"Mode={res_mode:.4f}")
ax3.set_title("Residual Distribution")
ax3.legend()
figures.append(fig3)

plt.show()

###############################################################################
# === Save Report ===
params_info = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Model": "MLP (Feedforward NN)",
    "Horizon": HORIZON,
    "Lag": LAG,
    "Hidden size": HIDDEN_SIZE,
    "Layers": NUM_LAYERS,
    "Dropout": DROPOUT,
    "Batch size": BATCH_SIZE,
    "Learning rate": LR,
    "Epochs": EPOCHS,
    "Patience": PATIENCE,
    "R² (t+1)": float(f"{r2:.6f}"),
    "MAE (t+1)": float(f"{mae:.6f}"),
    "RMSE (t+1)": float(f"{rmse:.6f}"),
    "Residual mean": float(f"{res_mean:.6f}"),
    "Residual median": float(f"{res_median:.6f}"),
    "Residual mode": float(f"{res_mode:.6f}")
}

errors = [(target_item, r2, mae, rmse)]

report_file = f"mlp_parameter_selection_multi_output_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf"
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_report_to_pdf(
    filename=report_path,
    title=f"MLP Multi-output Forecast Report - {station_name}",
    params=params_info,
    features=features,
    errors=errors,
    figures=figures
)

print(f"\n✅ Report saved as {report_path}")
