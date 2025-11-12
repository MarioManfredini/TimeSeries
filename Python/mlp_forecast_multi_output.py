# -*- coding: utf-8 -*-
"""
Created 2025/11/0
MLP 24-hour multi-step forecasting and report generation
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

MODEL_PATH = "mlp_best_model_multi_output.keras"
PARAMS_PATH = "mlp_best_params_multi_output.json"
FEATURES_PATH = "mlp_features_used_multi_output.json"

###############################################################################
def save_mlp_formula_as_jpg(filename="formula_mlp.jpg"):
    """
    Save MLP model formula and cost function as a JPEG image with explanation.
    """
    formula_eq = (
        r"$h^{(1)} = \sigma(W^{(1)} x + b^{(1)})$" "\n"
        r"$h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})$" "\n"
        r"$\hat{y} = W^{(L)} h^{(L-1)} + b^{(L)}$" "\n"
        r"$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$"
    )

    explanation_lines = [
        r"$x$: input feature vector",
        r"$h^{(l)}$: hidden layer activations at layer $l$",
        r"$W^{(l)}, b^{(l)}$: learnable weights and biases at layer $l$",
        r"$\sigma$: non-linear activation (ReLU, tanh, etc.)",
        r"$\hat{y}$: output vector (multi-step predictions)",
        r"$\mathcal{L}$: mean squared error minimized during training",
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    ax.text(0, 1, formula_eq, fontsize=18, ha="left", va="top", linespacing=1.6)

    y_start = 0.4
    line_spacing = 0.06
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * line_spacing, line,
                fontsize=11, ha="left", va="top")

    plt.tight_layout()
    temp_file = "_temp_formula_mlp.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved MLP formula image as {filename}")

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Load Saved Parameters and Features (from training phase) ===
with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    best_params = json.load(f)

LAG = best_params.get("lag", 24)
FORECAST_HORIZON = best_params.get("horizon", 24)

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    features = json.load(f)

print(f"✅ Loaded feature list from {FEATURES_PATH} ({len(features)} features)")
print(f"✅ Forecast horizon set to {FORECAST_HORIZON}")

###############################################################################
# === Feature Engineering ===
df_features = {}

def add_feature_if_used(df, name, series):
    if name in features:
        df_features[name] = series

for item in ["NO(ppm)", "NO2(ppm)", "U", "V"]:
    if item in features:
        df_features[item] = df[item]

for item in ["Ox(ppm)", "NO(ppm)", "NO2(ppm)", "U", "V"]:
    col_mean = f"{item}_roll_mean_3"
    col_std = f"{item}_roll_std_6"
    add_feature_if_used(df, col_mean, df[item].shift(1).rolling(3).mean())
    add_feature_if_used(df, col_std, df[item].shift(1).rolling(6).std())

for item in ["Ox(ppm)"]:
    add_feature_if_used(df, f"{item}_diff_1", df[item].shift(1).diff(1))
    add_feature_if_used(df, f"{item}_diff_2", df[item].shift(1).diff(2))
    add_feature_if_used(df, f"{item}_diff_3", df[item].shift(1).diff(3))

for item in ["NO(ppm)", "NO2(ppm)", "U", "V"]:
    add_feature_if_used(df, f"{item}_diff_1", df[item].shift(1).diff(1))
    add_feature_if_used(df, f"{item}_diff_3", df[item].shift(1).diff(3))

add_feature_if_used(df, "hour_sin", np.sin(2 * np.pi * df["時"] / 24))
add_feature_if_used(df, "hour_cos", np.cos(2 * np.pi * df["時"] / 24))
add_feature_if_used(df, "dayofweek", df.index.dayofweek)
add_feature_if_used(df, "is_weekend", (df.index.dayofweek >= 5).astype(int))

df = pd.concat([df, pd.DataFrame(df_features, index=df.index)], axis=1)
df = df.loc[:, ~df.columns.duplicated()].copy()

###############################################################################
# === Multi-output Target creation ===
for i in range(FORECAST_HORIZON):
    df[f"{target_item}_t+{i+1:02}"] = df[target_item].shift(-i-1)

target_cols = [f"{target_item}_t+{i+1:02}" for i in range(FORECAST_HORIZON)]
data_model = df.dropna(subset=features + target_cols).copy()

###############################################################################
# === Normalization ===
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_array = scaler_X.fit_transform(data_model[features])
y_array = scaler_y.fit_transform(data_model[target_cols])

X_base = pd.DataFrame(X_array, columns=features)
y = pd.DataFrame(y_array, columns=target_cols)

###############################################################################
# === Create lag-expanded features to match model input_dim ===
# Each sample should contain features from the past LAG hours (e.g. 24)
X_lagged = []

for lag in range(LAG):
    shifted = X_base.shift(lag)
    shifted.columns = [f"{c}_lag{lag}" for c in features]
    X_lagged.append(shifted)

# Combine all lag blocks horizontally
X_full = pd.concat(X_lagged, axis=1)

# Drop NaN rows caused by shifting
data_lagged = pd.concat([X_full, y], axis=1).dropna()
X_full = data_lagged[X_full.columns]
y = data_lagged[target_cols]

###############################################################################
# === Train/Test Split ===
split_index = int(len(X_full) * 0.7)
X_train, X_test = X_full[:split_index], X_full[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

###############################################################################
# === Load Best Model ===
model = load_model(MODEL_PATH)
print(f"✅ Loaded trained MLP model from {MODEL_PATH}")

# Check input compatibility
if X_test.shape[1] != model.input_shape[-1]:
    raise ValueError(
        f"❌ Input mismatch: model expects {model.input_shape[-1]} features, "
        f"but X_test has {X_test.shape[1]}. "
        f"Check LAG and feature engineering."
    )

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
kde = gaussian_kde(residuals)
x_vals = np.linspace(min(residuals), max(residuals), 1000)
res_mode = x_vals[np.argmax(kde(x_vals))]

###############################################################################
# === Figures ===
figures = []

# Predicted vs True
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(y_true.flatten()[-720:], label='True values', color='gray')
ax1.plot(y_pred.flatten()[-720:], label='MLP forecast', color='lightgray', linestyle='dashed')
ax1.set_title(f"MLP Multi-step Forecast (24h)\nR² (avg): {np.mean(r2_list):.5f}")
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

# Histogram
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

# Multi-step comparison
steps = [f't+{i:02}' for i in range(1, FORECAST_HORIZON + 1)]
comparison_figs = plot_comparison_actual_predicted(y_true, y_pred, target_cols, steps, rows_per_page=3)
figures.extend(comparison_figs)

###############################################################################
# === Report Info ===
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_str = f"{int(elapsed_time//60)} min {int(elapsed_time%60)} sec"

formula_path = os.path.join("..", "reports", "formula_mlp.jpg")
save_mlp_formula_as_jpg(filename=formula_path)

explanation = (
    "The Multi-Layer Perceptron (MLP) is a feedforward neural network composed of\n"
    "an input layer, one or more hidden layers, and an output layer.\n\n"
    "Each layer applies a linear transformation followed by a non-linear activation,\n"
    "allowing the network to model complex, non-linear relationships.\n\n"
    "In this application, the MLP outputs 24 simultaneous predictions (multi-output)\n"
    "corresponding to the next 24 hours of Ox(ppm) concentrations.\n\n"
    "The model is trained by minimizing the mean squared error between predicted\n"
    "and observed values."
)

params_info = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Number of data points in the train set": len(y_train),
    "Number of data points in the test set": len(y_test),
    "Forecast horizon (hours)": FORECAST_HORIZON,
    "Model": "MLP",
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
report_file = f"mlp_forecast_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf"
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_report_to_pdf(
    filename=report_path,
    title=f"MLP 24-hour Forecast Report - {station_name}",
    formula_image_path=formula_path,
    comments=explanation,
    params=params_info,
    features=features,
    errors=errors,
    figures=figures
)

print(f"\n✅ Report saved as {report_path}")
