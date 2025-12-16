# -*- coding: utf-8 -*-
"""
Created 2025/12/14
SARIMAX 24h iterative forecasting – Report Generator
@author: Mario
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utility import load_and_prepare_data, get_station_name
from report import save_report_to_pdf, plot_comparison_actual_predicted, plot_error_summary_page

warnings.filterwarnings("ignore")

start_time = time.time()

###############################################################################
# === Parameters ===
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)

target_item = 'Ox(ppm)'
forecast_horizon = 24

# Best parameters found previously
order = (2, 0, 0)
seasonal_order = (0, 0, 0, 24)

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Feature Engineering ===
# Time features only (only features that can be used for future exogenous values)
df["hour_sin"] = np.sin(2 * np.pi * df["時"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["時"] / 24)
df["dayofweek"] = df.index.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
features = ['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend']

# Drop NaNs
data_model = df.dropna(subset=features + [target_item]).copy()

###############################################################################
X = data_model[features]
y = data_model[target_item]

# === Train/Test Split (same logic as before) ===
TRAIN_DAYS = 90
TEST_DAYS = 30

hours_per_day = 24
train_hours = TRAIN_DAYS * hours_per_day
test_hours = TEST_DAYS * hours_per_day

X_train = X.iloc[-(train_hours + test_hours):-test_hours]
X_test = X.iloc[-test_hours:]

y_train = y.iloc[-(train_hours + test_hours):-test_hours]
y_test = y.iloc[-test_hours:]

###############################################################################
# === Scale exogenous variables (correct way) ===
scaler_X = MinMaxScaler()

X_train_scaled = pd.DataFrame(
    scaler_X.fit_transform(X_train),
    index=X_train.index,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    scaler_X.transform(X_test),
    index=X_test.index,
    columns=X_test.columns
)

###############################################################################
y_preds = []
y_trues = []

n_test = len(X_test_scaled)

for start in range(n_test - forecast_horizon):

    # ============================
    # 1) Expanding training window
    # ============================
    y_train_roll = pd.concat([
        y_train,
        y_test.iloc[:start]
    ])

    X_train_roll = pd.concat([
        X_train_scaled,
        X_test_scaled.iloc[:start]
    ])

    # ============================
    # 2) Fit model on rolling data
    # ============================
    model = SARIMAX(
        y_train_roll,
        exog=X_train_roll,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=True
    )

    model_fit = model.fit(disp=False)

    # ============================
    # 3) Future known exogenous vars
    # ============================
    exog_future = X_test_scaled.iloc[start:start + forecast_horizon]

    # ============================
    # 4) Multi-step forecast
    # ============================
    forecast = model_fit.forecast(
        steps=forecast_horizon,
        exog=exog_future
    )

    # ============================
    # 5) Store results
    # ============================
    y_preds.append(forecast.values)
    y_trues.append(
        y_test.iloc[start:start + forecast_horizon].values
    )

    print(f"- Rolling step {start+1}/{n_test - forecast_horizon}")

y_preds = np.array(y_preds)   # shape (n_samples, 24)
y_trues = np.array(y_trues)

###############################################################################
# === Evaluation per horizon ===
r2_list, mae_list, rmse_list = [], [], []

for h in range(forecast_horizon):
    y_true_h = y_trues[:, h]
    y_pred_h = y_preds[:, h]

    r2_list.append(r2_score(y_true_h, y_pred_h))
    mae_list.append(mean_absolute_error(y_true_h, y_pred_h))
    rmse_list.append(np.sqrt(mean_squared_error(y_true_h, y_pred_h)))

    print(
        f"t+{h+1:02d} "
        f"R²={r2_list[-1]:.4f} "
        f"MAE={mae_list[-1]:.4f} "
        f"RMSE={rmse_list[-1]:.4f}"
    )

###############################################################################
# === Plots (correct multi-step alignment) ===
figures = []

steps = [f"t+{i:02d}" for i in range(1, forecast_horizon + 1)]

figs = plot_comparison_actual_predicted(
    y_trues,
    y_preds,
    target_cols=steps,
    steps=steps,
    rows_per_page=3
)
figures.extend(figs)

figs = plot_error_summary_page(mae_list, rmse_list, r2_list, steps)
figures.append(figs)

###############################################################################
# === Report ===
elapsed = int(time.time() - start_time)

params = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Model": "SARIMAX",
    "Order (p,d,q)": order,
    "Seasonal order": seasonal_order,
    "Forecast horizon": forecast_horizon,
    "Training samples": len(y_train),
    "Test samples": len(y_test),
    "Elapsed time": f"{elapsed // 60} min {elapsed % 60} sec",
}

errors = [
    (steps[i], r2_list[i], mae_list[i], rmse_list[i])
    for i in range(forecast_horizon)
]

report_path = os.path.join(
    "..", "reports",
    f"sarimax_24h_forecast_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf"
)

save_report_to_pdf(
    filename=report_path,
    title=f"SARIMAX 24h Forecast Report – {station_name}",
    params=params,
    features=features,
    errors=errors,
    figures=figures
)

print(f"\n✅ SARIMAX 24h forecast report saved: {report_path}")
