# -*- coding: utf-8 -*-
"""
Created 2025/11/03
Linear Regression 24-hour multi-step forecasting and report generation
Author: Mario
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import gaussian_kde
from PIL import Image

from utility import get_station_name, load_and_prepare_data
from report import save_report_to_pdf, plot_comparison_actual_predicted

###############################################################################
# === Parameters ===
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'
LAG = 24
FORECAST_HORIZON = 24
MODEL_PATH = "linear_best_model_multi_output.pkl"

start_time = time.time()

###############################################################################
def save_linear_formula_as_jpg(filename="formula_linear.jpg"):
    """
    Save Linear Regression formula and cost function as a JPEG image.
    """
    formula_eq = (
        r"$\hat{y}_t = \beta_0 + \sum_{i=1}^{n} \beta_i x_{t,i}$" "\n"
        r"$\mathcal{L} = \frac{1}{N} \sum_{t=1}^{N} (y_t - \hat{y}_t)^2$"
    )

    explanation_lines = [
        r"$x_{t,i}$: i-th feature value at time t",
        r"$\beta_i$: regression coefficient learned from training data",
        r"$\beta_0$: intercept term (bias)",
        r"$\hat{y}_t$: predicted target value",
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
    temp_file = "_temp_formula_linear.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved Linear Regression formula image as {filename}")

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Feature Engineering ===
features = [
    "NO(ppm)",
    "NO2(ppm)",
    "U",
    "V",
    "Ox(ppm)_roll_mean_3",
    "Ox(ppm)_roll_std_6",
    "NO(ppm)_roll_mean_3",
    "NO(ppm)_roll_std_6",
    "NO2(ppm)_roll_mean_3",
    "NO2(ppm)_roll_std_6",
    "U_roll_mean_3",
    "U_roll_std_6",
    "V_roll_mean_3",
    "V_roll_std_6",
    "Ox(ppm)_diff_1",
    "Ox(ppm)_diff_2",
    "Ox(ppm)_diff_3",
    "NO(ppm)_diff_3",
    "NO2(ppm)_diff_3",
    "U_diff_3",
    "V_diff_3",
    "hour_sin",
    "hour_cos",
    "dayofweek",
    "is_weekend"
]
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

X = pd.DataFrame(X_array, columns=features)
y = pd.DataFrame(y_array, columns=target_cols)

###############################################################################
# === Train/Test Split ===
split_index = int(len(X) * 0.7)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

###############################################################################
# === Load or Train Model ===
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded saved Linear Regression model from {MODEL_PATH}")
else:
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Trained and saved new Linear Regression model to {MODEL_PATH}")

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
ax1.plot(y_pred.flatten()[-720:], label='Linear forecast', color='lightgray', linestyle='dashed')
ax1.set_title(f"Linear Regression Multi-step Forecast (24h)\nR² (avg): {np.mean(r2_list):.5f}")
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

formula_path = os.path.join("..", "reports", "formula_linear.jpg")
save_linear_formula_as_jpg(filename=formula_path)

explanation = (
    "The Linear Regression model establishes a direct linear relationship between\n"
    "the input features and the target variable(s).\n\n"
    "Each forecasted value is computed as a weighted sum of the input features\n"
    "plus a bias term. The model coefficients (weights) are estimated by minimizing\n"
    "the Mean Squared Error between predicted and actual values.\n\n"
    "This method is fast, interpretable, and effective when the relationship between\n"
    "predictors and target is approximately linear."
)

params_info = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Number of data points in the train set": len(y_train),
    "Number of data points in the test set": len(y_test),
    "Forecast horizon (hours)": FORECAST_HORIZON,
    "Model": "Linear Regression",
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
report_file = f"linear_forecast_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf"
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_report_to_pdf(
    filename=report_path,
    title=f"Linear Regression 24-hour Forecast Report - {station_name}",
    formula_image_path=formula_path,
    comments=explanation,
    params=params_info,
    features=features,
    errors=errors,
    figures=figures
)

print(f"\n✅ Report saved as {report_path}")
