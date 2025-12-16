# -*- coding: utf-8 -*-
"""
Created 2025/11/16

LightGBM Multi-Output (24h Forecast) – Report Generator
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
from lightgbm import LGBMRegressor

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

n_estimators = 450
max_depth = 12
learning_rate = 0.05
min_child_samples = 6
colsample_bytree = 1.0
subsample = 0.01
num_leaves = 32

###############################################################################
# === Formula Image (LightGBM) ===
def save_lightgbm_formula_as_jpg(filename="formula_lgbm.jpg"):

    formula_gain = (
        r"$\text{Gain}(f) = \frac{1}{2}\left["
        r"\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}"
        r"\right] - \gamma$"
    )

    explanation_lines = [
        r"LightGBM builds trees using Gradient Boosting.",
        r"At each split, it selects the feature maximizing the Gain.",
        r"G is the gradient sum, H is the Hessian sum.",
        r"λ and γ are regularization parameters controlling leaf weight and tree structure.",
        r"Multi-output prediction is handled by fitting one model per output dimension."
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis("off")

    ax.text(0, 1, formula_gain, fontsize=19, ha="left", va="top")

    y_start = 0.72
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * 0.08, line, fontsize=12, ha="left", va="top")

    temp_file = "_temp_formula_lgbm.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved LightGBM formula image as {filename}")


###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Feature Engineering ===
features = ['NO(ppm)', 'NO2(ppm)', 'U', 'V']
lagged_items = [target_item, 'NO(ppm)', 'NO2(ppm)', 'U', 'V']

# Lag features
for item in lagged_items:
    for l in range(1, 24):
        df[f'{item}_lag{l}'] = df[item].shift(l)
        features.append(f'{item}_lag{l}')

# Rolling stats
for item in lagged_items:
    df[f'{item}_roll_mean_3'] = df[item].shift(1).rolling(3).mean()
    df[f'{item}_roll_std_6'] = df[item].shift(1).rolling(6).std()
    features += [f'{item}_roll_mean_3', f'{item}_roll_std_6']

# Differences
df[f'{target_item}_diff_1'] = df[target_item].shift(1).diff(1)
df[f'{target_item}_diff_2'] = df[target_item].shift(1).diff(2)
df[f'{target_item}_diff_3'] = df[target_item].shift(1).diff(3)
features += [f'{target_item}_diff_1', f'{target_item}_diff_2', f'{target_item}_diff_3']

for item in ['NO(ppm)', 'NO2(ppm)', 'U', 'V']:
    df[f'{item}_diff_3'] = df[item].shift(1).diff(3)
    features.append(f'{item}_diff_3')

# Time features
df["hour_sin"] = np.sin(2 * np.pi * df["時"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["時"] / 24)
df["dayofweek"] = df.index.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
features += ['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend']

# Multi-output target
for i in range(forecast_horizon):
    df[f'{target_item}_t+{i+1:02}'] = df[target_item].shift(-i-1)

target_cols = [f'{target_item}_t+{i+1:02}' for i in range(forecast_horizon)]
data_model = df.dropna(subset=features + target_cols).copy()

###############################################################################
# === Train/Test Split ===
TRAIN_DAYS = 650
TEST_DAYS = 280

hours_per_day = 24
train_hours = TRAIN_DAYS * hours_per_day
test_hours = TEST_DAYS * hours_per_day

X_raw = data_model[features]
y_raw = data_model[target_cols]

X_train_raw = X_raw.iloc[-(train_hours + test_hours):-test_hours]
X_test_raw  = X_raw.iloc[-test_hours:]

y_train = y_raw.iloc[-(train_hours + test_hours):-test_hours]
y_test  = y_raw.iloc[-test_hours:]

###############################################################################
# === Normalization (INPUT ONLY, no leakage) ===
scaler_X = MinMaxScaler()
X_train = pd.DataFrame(
    scaler_X.fit_transform(X_train_raw),
    columns=features,
    index=X_train_raw.index
)

X_test = pd.DataFrame(
    scaler_X.transform(X_test_raw),
    columns=features,
    index=X_test_raw.index
)

###############################################################################
# === Train LightGBM Multi-Output ===
base_model = LGBMRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_child_samples=min_child_samples,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    num_leaves=num_leaves,
    random_state=42
)

model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

###############################################################################
# === Predictions (no inverse transform needed) ===
y_pred = model.predict(X_test)
y_true = y_test.values

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
formula_path = os.path.join("..", "reports", "formula_lgbm.jpg")
save_lightgbm_formula_as_jpg(filename=formula_path)

###############################################################################
# === Report parameters ===
end_time = time.time()
elapsed_seconds = int(end_time - start_time)
elapsed_time_str = f"{elapsed_seconds // 60} min {elapsed_seconds % 60} sec"

explanation = (
    "LightGBM uses Gradient Boosting over decision trees.\n"
    "Each tree is constructed by choosing the split that maximizes the Gain.\n"
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
    "Model": "LightGBM (MultiOutput)",
    "Elapsed time": elapsed_time_str,
    "Number of estimators": n_estimators,
    "Max depth": max_depth,
    "Learning rate": learning_rate,
    "Min child samples": min_child_samples,
    "Column sample by tree": colsample_bytree,
    "Subsample": subsample,
    "Number of leaves": num_leaves,
}

errors = [
    (target_cols[i], r2_list[i], mae_list[i], rmse_list[i])
    for i in range(forecast_horizon)
]

###############################################################################
# === Save Report ===
report_file = (
    f'lgbm_forecast_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf'
)
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_report_to_pdf(
    filename=report_path,
    title=f"LightGBM Forecasting Report - {station_name}",
    formula_image_path=formula_path,
    comments=explanation,
    params=model_params,
    features=features,
    errors=errors,
    figures=figures,
)

print(f"\n✅ Report saved as {report_path}")
