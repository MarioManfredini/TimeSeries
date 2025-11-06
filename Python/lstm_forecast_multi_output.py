# -*- coding: utf-8 -*-
"""
Created 2025/11/03
LSTM 24-hour multi-step forecasting and report generation
Author: Mario
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import gaussian_kde
from PIL import Image

from utility import get_station_name, load_and_prepare_data
from report import save_report_to_pdf, plot_comparison_actual_predicted

start_time = time.time()

###############################################################################
# === Parameters ===
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'
MODEL_PATH = "lstm_best_model_multi_output.keras"
PARAMS_PATH = "lstm_best_params_multi_output.json"
FEATURES_PATH = "lstm_features_used_multi_output.json"

###############################################################################
def save_lstm_formula_as_jpg(filename="formula_lstm.jpg"):
    """
    Save LSTM model formula and cost function as a JPEG image with explanation.
    """
    # --- LSTM main equations (using only matplotlib-compatible LaTeX) ---
    formula_eq = (
        r"$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$" "\n"
        r"$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$" "\n"
        r"$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$" "\n"
        r"$\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$" "\n"
        r"$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$" "\n"
        r"$h_t = o_t \odot \tanh(c_t)$" "\n"
        r"$\hat{y}_t = W_y h_t + b_y$"
    )

    # --- Cost function ---
    formula_cost = (
        r"$\mathcal{L} = \frac{1}{N} \sum_{t=1}^{N} (y_t - \hat{y}_t)^2$"
    )

    # --- Explanation text ---
    explanation_lines = [
        r"$i_t, f_t, o_t$: input, forget and output gates controlling memory flow",
        r"$c_t$: cell state storing long-term information",
        r"$h_t$: hidden state passed to the next time step",
        r"$\sigma$: sigmoid activation function",
        r"$\tanh$: hyperbolic tangent activation for non-linearity",
        r"$\mathcal{L}$: mean squared error minimized during training",
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    # Draw equations
    ax.text(0, 1, formula_eq, fontsize=16, ha="left", va="top", linespacing=1.5)
    ax.text(0, 0.5, formula_cost, fontsize=18, ha="left", va="top")

    # Draw explanation lines
    y_start = 0.25
    line_spacing = 0.07
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * line_spacing, line,
                fontsize=11, ha="left", va="top")

    plt.tight_layout()

    # Save as temporary PNG
    temp_file = "_temp_formula_lstm.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Convert to JPEG
    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved LSTM formula image as {filename}")

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Load Saved Parameters and Features (from training phase) ===
with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    best_params = json.load(f)

LAG = best_params.get("lag", 24)
FORECAST_HORIZON = best_params.get("horizon", 1)

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    features = json.load(f)

print(f"✅ Loaded feature list from {FEATURES_PATH} ({len(features)} features)")
print(f"✅ Forecast horizon set to {FORECAST_HORIZON}")

###############################################################################
# === Feature Engineering (replica coerente con il training) ===
# Si ricostruiscono solo le feature presenti nella lista salvata
df_features = {}

# Funzione di supporto per calcolare in modo dinamico le feature necessarie
def add_feature_if_used(df, name, series):
    if name in features:
        df_features[name] = series

# Base pollutants
for item in ["NO(ppm)", "NO2(ppm)", "U", "V"]:
    if item in features:
        df_features[item] = df[item]

# Rolling mean/std
for item in ["Ox(ppm)", "NO(ppm)", "NO2(ppm)", "U", "V"]:
    col_mean = f"{item}_roll_mean_3"
    col_std = f"{item}_roll_std_6"
    add_feature_if_used(df, col_mean, df[item].shift(1).rolling(3).mean())
    add_feature_if_used(df, col_std, df[item].shift(1).rolling(6).std())

# Differenze
for item in ["Ox(ppm)"]:
    add_feature_if_used(df, f"{item}_diff_1", df[item].shift(1).diff(1))
    add_feature_if_used(df, f"{item}_diff_2", df[item].shift(1).diff(2))
    add_feature_if_used(df, f"{item}_diff_3", df[item].shift(1).diff(3))

for item in ["NO(ppm)", "NO2(ppm)", "U", "V"]:
    add_feature_if_used(df, f"{item}_diff_3", df[item].shift(1).diff(3))

# Variabili temporali
add_feature_if_used(df, "hour_sin", np.sin(2 * np.pi * df["時"] / 24))
add_feature_if_used(df, "hour_cos", np.cos(2 * np.pi * df["時"] / 24))
add_feature_if_used(df, "dayofweek", df.index.dayofweek)
add_feature_if_used(df, "is_weekend", (df.index.dayofweek >= 5).astype(int))

# Combina le feature calcolate
df = pd.concat([df, pd.DataFrame(df_features, index=df.index)], axis=1)
df = df.loc[:, ~df.columns.duplicated()].copy()

###############################################################################
# === Multi-output Target creation ===
# Crea le colonne target in base all’orizzonte salvato nel file di parametri
for i in range(FORECAST_HORIZON):
    df[f"{target_item}_t+{i+1:02}"] = df[target_item].shift(-i-1)

target_cols = [f"{target_item}_t+{i+1:02}" for i in range(FORECAST_HORIZON)]

# Drop di righe con valori NaN
data_model = df.dropna(subset=features + target_cols).copy()

###############################################################################
# === Normalization ===
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

print("🔎 len(features):", len(features))
print("🔎 data_model[features].shape:", data_model[features].shape)
print("🔎 data_model.columns:", list(data_model.columns))

X_array = scaler_X.fit_transform(data_model[features])
y_array = scaler_y.fit_transform(data_model[target_cols])

print("✅ X_array shape:", X_array.shape)

X = pd.DataFrame(X_array, columns=features)
y = pd.DataFrame(y_array, columns=target_cols)

###############################################################################
# === Train/Test Split ===
split_index = int(len(X) * 0.7)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Reshape for LSTM: (samples, timesteps, features)
# Convert DataFrame to NumPy array before reshaping
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

###############################################################################
# === Load Best Model ===
model = load_model(MODEL_PATH)
print(f"✅ Loaded trained LSTM model from {MODEL_PATH}")

###############################################################################
# === Forecast ===
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

###############################################################################
# === Evaluation ===
r2_list, mae_list, rmse_list = [], [], []

for i, col in enumerate(target_cols):
    r2 = r2_score(y_true[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    r2_list.append(r2)
    mae_list.append(mae)
    rmse_list.append(rmse)
    print(f"{col}: R²={r2:.4f}, MAE={mae:.5f}, RMSE={rmse:.5f}")

###############################################################################
# === Residual Analysis ===
residuals = y_true.flatten() - y_pred.flatten()
res_mean = np.mean(residuals)
res_median = np.median(residuals)
# Estimate mode using KDE
kde = gaussian_kde(residuals)
x_vals = np.linspace(min(residuals), max(residuals), 1000)
res_mode = x_vals[np.argmax(kde(x_vals))]

###############################################################################
# === Figures ===
figures = []

# Predicted vs True (flattened for visualization)
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(y_true.flatten()[-720:], label='True values', color='gray')
ax1.plot(y_pred.flatten()[-720:], label='LSTM forecast', color='lightgray', linestyle='dashed')
ax1.set_title(f"LSTM Multi-step Forecast (24h)\nR² (avg): {np.mean(r2_list):.5f}")
ax1.set_xlabel("Samples")
ax1.set_ylabel(target_item)
ax1.legend()
ax1.grid(True)
fig1.tight_layout()
figures.append(fig1)

# Residual scatter
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.scatter(range(len(residuals)), residuals, alpha=0.4, s=10, color='gray')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_title("Distribution of Residual Errors")
ax2.set_xlabel("Samples")
ax2.set_ylabel("Residuals (True - Predicted)")
ax2.grid(True)
fig2.tight_layout()
figures.append(fig2)

# Histogram of residuals
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.hist(residuals, bins=50, color='lightgray', edgecolor='gray', alpha=0.8)
ax3.axvline(res_mean, color='black', linestyle='--', linewidth=1, label=f"Mean = {res_mean:.6f}")
ax3.axvline(res_median, color='black', linestyle='-.', linewidth=1, label=f"Median = {res_median:.6f}")
ax3.axvline(res_mode, color='black', linestyle=':', linewidth=1, label=f"Mode = {res_mode:.6f}")
ax3.legend()
ax3.set_title("Histogram of Residuals – Distribution & Central Tendency")
ax3.set_xlabel("Residual value")
ax3.set_ylabel("Frequency")
ax3.grid(True)
fig3.tight_layout()
figures.append(fig3)

# Multi-step comparison plots
steps = [f't+{i:02}' for i in range(1, FORECAST_HORIZON + 1)]
comparison_figs = plot_comparison_actual_predicted(y_true, y_pred, target_cols, steps, rows_per_page=3)
figures.extend(comparison_figs)

###############################################################################
# === Report Info ===
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_str = f"{int(elapsed_time//60)} min {int(elapsed_time%60)} sec"

# Save formula as image
formula_path = os.path.join("..", "reports", "formula_lstm.jpg")
save_lstm_formula_as_jpg(filename=formula_path)

explanation = (
    "The LSTM (Long Short-Term Memory) model is a type of recurrent neural network (RNN)\n"
    "designed to learn temporal dependencies in sequential data. It introduces a memory cell\n"
    "that can preserve information across long time intervals through a system of gates:\n\n"
    "1. The **input gate** decides how much new information enters the memory cell.\n"
    "2. The **forget gate** controls what information should be discarded from the cell state.\n"
    "3. The **output gate** determines how much of the internal memory contributes to the output.\n\n"
    "This gating mechanism allows the LSTM to model both short-term and long-term dependencies.\n"
    "During training, the model minimizes the mean squared error between predicted and actual values."
)

params_info = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Number of data points in the train set": len(y_train),
    "Number of data points in the test set": len(y_test),
    "Forecast horizon (hours)": FORECAST_HORIZON,
    "Model": "LSTM",
    "Elapsed time": elapsed_str,
    "Number of features used": len(features),
    "Residuals mean": float(f"{res_mean:.6f}"),
    "Residuals median": float(f"{res_median:.6f}"),
    "Residuals mode": float(f"{res_mode:.6f}"),
}

errors = [
    (target_cols[i], r2_list[i], mae_list[i], rmse_list[i])
    for i in range(FORECAST_HORIZON)
]

###############################################################################
# === Save Report ===
report_file = f"lstm_forecast_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf"
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_report_to_pdf(
    filename=report_path,
    title=f"LSTM 24-hour Forecast Report - {station_name}",
    formula_image_path=formula_path,
    comments=explanation,
    params=params_info,
    features=features,
    errors=errors,
    figures=figures
)

print(f"\n✅ Report saved as {report_path}")
