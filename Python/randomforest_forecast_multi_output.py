# -*- coding: utf-8 -*-
"""
Created 2025/11/16

RandomForest Multi-Output (24h Forecast) – Report Generator
@author: Mario
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from utility import get_station_name, load_and_prepare_data
from report import save_report_to_pdf, plot_feature_summary_pages, plot_comparison_actual_predicted

start_time = time.time()

###############################################################################
# === Parameters ===
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'
forecast_horizon = 24

# Best parameters
BEST_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 20,
    "min_samples_split": 3,
    "min_samples_leaf": 1,
    "max_features": "sqrt"
}

###############################################################################
# === Formula Image (Random Forest) ===
def save_randomforest_formula_as_jpg(filename="formula_randomforest.jpg"):

    formula = (
        r"$\hat{y}(x) = \frac{1}{M} \sum_{m=1}^{M} T_m(x)$"
    )

    explanation_lines = [
        r"Random Forest builds an ensemble of decision trees.",
        r"Each tree is trained on a bootstrap sample of the data.",
        r"At each split, a random subset of features is considered (max_features).",
        r"The prediction is the average over all trees.",
        r"Depth and leaf parameters control overfitting vs generalization."
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis("off")

    ax.text(0, 1, formula, fontsize=22, ha="left", va="top")
    y_start = 0.55
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * 0.08, line, fontsize=12, ha="left", va="top")

    temp_file = "_temp_formula_randomforest.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved Random Forest formula image as {filename}")

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Feature Engineering ===
features = ['NO(ppm)', 'NO2(ppm)', 'U', 'V']
lagged_items = [target_item, 'NO(ppm)', 'NO2(ppm)', 'U', 'V']

for item in lagged_items:
    for l in range(1, 24):
        df[f'{item}_lag{l}'] = df[item].shift(l)
        features.append(f'{item}_lag{l}')

for item in lagged_items:
    df[f'{item}_roll_mean_3'] = df[item].shift(1).rolling(3).mean()
    df[f'{item}_roll_std_6'] = df[item].shift(1).rolling(6).std()
    features += [f'{item}_roll_mean_3', f'{item}_roll_std_6']

df[f'{target_item}_diff_1'] = df[target_item].shift(1).diff(1)
df[f'{target_item}_diff_2'] = df[target_item].shift(1).diff(2)
df[f'{target_item}_diff_3'] = df[target_item].shift(1).diff(3)
features += [f'{target_item}_diff_1', f'{target_item}_diff_2', f'{target_item}_diff_3']

for item in ['NO(ppm)', 'NO2(ppm)', 'U', 'V']:
    df[f'{item}_diff_3'] = df[item].shift(1).diff(3)
    features.append(f'{item}_diff_3')

df["hour_sin"] = np.sin(2 * np.pi * df["時"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["時"] / 24)
df["dayofweek"] = df.index.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
features += ['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend']

for i in range(forecast_horizon):
    df[f'{target_item}_t+{i+1:02}'] = df[target_item].shift(-i-1)

target_cols = [f'{target_item}_t+{i+1:02}' for i in range(forecast_horizon)]
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
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

###############################################################################
# === Train Random Forest Multi-Output ===
print("Training Random Forest model...")

base_model = RandomForestRegressor(
    n_estimators=BEST_PARAMS["n_estimators"],
    max_depth=BEST_PARAMS["max_depth"],
    min_samples_split=BEST_PARAMS["min_samples_split"],
    min_samples_leaf=BEST_PARAMS["min_samples_leaf"],
    max_features=BEST_PARAMS["max_features"],
    n_jobs=-1,
    random_state=42
)

model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

###############################################################################
# === Predictions ===
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

###############################################################################
# === Errors per step ===
print("Error per forecast step:")
r2_list, mae_list, rmse_list = [], [], []

for i, col in enumerate(target_cols):
    r2s = r2_score(y_true[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    r2_list.append(r2s)
    mae_list.append(mae)
    rmse_list.append(rmse)
    print(f"{col}: R²={r2s:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

###############################################################################
# === Plots ===
figures = []
steps = [f't+{i:02}' for i in range(1, forecast_horizon + 1)]

figs = plot_comparison_actual_predicted(y_true, y_pred, target_cols, steps, rows_per_page=3)
figures.extend(figs)

figs = plot_feature_summary_pages(model, X_train.columns, target_cols,
                                  mae_list, rmse_list, r2_list, steps,
                                  top_n=20)
figures.extend(figs)

###############################################################################
# === Formula Image ===
formula_path = os.path.join("..", "reports", "formula_randomforest.jpg")
save_randomforest_formula_as_jpg(filename=formula_path)

###############################################################################
# === Report parameters ===
end_time = time.time()
elapsed_seconds = int(end_time - start_time)
elapsed_time_str = f"{elapsed_seconds // 60} min {elapsed_seconds % 60} sec"

explanation = (
    "Random Forest averages predictions from multiple decision trees.\n"
    "Each tree is trained on a bootstrap sample of the data.\n"
    "At each split, a random subset of features is considered (max_features).\n"
    "The model produces a separate prediction for each forecast horizon (1–24 hours).\n"
    "Feature importances reflect the contribution to reduction in prediction error."
)

model_params = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Forecast horizon (hours)": forecast_horizon,
    "Training samples": len(y_train),
    "Test samples": len(y_test),
    "Features used": len(features),
    "Model": "RandomForest (MultiOutput)",
    "Elapsed time": elapsed_time_str,
}

errors = [
    (target_cols[i], r2_list[i], mae_list[i], rmse_list[i])
    for i in range(forecast_horizon)
]

###############################################################################
# === Save Report ===
report_file = (
    f'randomforest_forecast_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf'
)
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_report_to_pdf(
    filename=report_path,
    title=f"Random Forest Forecasting Report - {station_name}",
    formula_image_path=formula_path,
    comments=explanation,
    params=model_params,
    features=features,
    errors=errors,
    figures=figures,
)

print(f"\n✅ Report saved as {report_path}")
