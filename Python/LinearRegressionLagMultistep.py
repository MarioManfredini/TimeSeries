# -*- coding: utf-8 -*-
"""
Created 2025/06/08
Adapted from LightGBM script for multi-step Linear Regression
"""

import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utility import get_station_name, load_and_prepare_data
from report import save_report_to_pdf, plot_feature_summary_pages, plot_comparison_actual_predicted
import time

start_time = time.time()

###############################################################################
# === Parameters ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'
forecast_horizon = 24  # n-step forecast

###############################################################################
# === Load Data ===
data, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Feature Engineering ===
lags = 24
features = ['NO(ppm)', 'NO2(ppm)', 'U', 'V']
lagged_items = [target_item, 'NO(ppm)', 'NO2(ppm)', 'U', 'V']

lag_features = {}
for item in lagged_items:
    for l in range(1, lags):
        col_name = f'{item}_lag{l}'
        lag_features[col_name] = data[item].shift(l)
        features.append(col_name)

data = pd.concat([data, pd.DataFrame(lag_features, index=data.index)], axis=1)

rolling_features = {}
for item in lagged_items:
    col_mean = f'{item}_roll_mean_3'
    col_std = f'{item}_roll_std_6'
    rolling_features[col_mean] = data[item].shift(1).rolling(3).mean()
    rolling_features[col_std] = data[item].shift(1).rolling(6).std()
    features += [col_mean, col_std]

data = pd.concat([data, pd.DataFrame(rolling_features, index=data.index)], axis=1)

diff_features = {}
diff_features[f'{target_item}_diff_1'] = data[target_item].shift(1).diff(1)
diff_features[f'{target_item}_diff_2'] = data[target_item].shift(1).diff(2)
diff_features[f'{target_item}_diff_3'] = data[target_item].shift(1).diff(3)
features += list(diff_features.keys())

for item in ['NO(ppm)', 'NO2(ppm)', 'U', 'V']:
    col_name = f'{item}_diff_3'
    diff_features[col_name] = data[item].shift(1).diff(3)
    features.append(col_name)

data = pd.concat([data, pd.DataFrame(diff_features, index=data.index)], axis=1)

data["hour_sin"] = np.sin(2 * np.pi * data["時"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["時"] / 24)
data["dayofweek"] = data.index.dayofweek
data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)
features += ['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend']

# Multi-step target
for i in range(forecast_horizon):
    data[f'{target_item}_t+{i+1:02}'] = data[target_item].shift(-i-1)
target_cols = [f'{target_item}_t+{i+1:02}' for i in range(forecast_horizon)]

data_model = data.dropna(subset=features + target_cols).copy()

###############################################################################
# === Normalization ===
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(data_model[features])
y = scaler_y.fit_transform(data_model[target_cols])

X = pd.DataFrame(X, columns=features)
y = pd.DataFrame(y, columns=target_cols)

###############################################################################
# === Train/Test Split ===
split_index = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

###############################################################################
# === Model Training ===
base_model = LinearRegression()
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

###############################################################################
# === Forecast ===
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

###############################################################################
# === Evaluation ===
print("Error per forecast step (t+01, t+02, ...):")
r2_list, mae_list, rmse_list = [], [], []
for i, col in enumerate(target_cols):
    r2 = r2_score(y_true[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    r2_list.append(r2)
    mae_list.append(mae)
    rmse_list.append(rmse)
    print(f"{col}: R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

###############################################################################
# === Report Figures ===
figures = []

steps = [f't+{i:02}' for i in range(1, forecast_horizon + 1)]

figs = plot_comparison_actual_predicted(y_true, y_pred, target_cols, steps, rows_per_page=3)
figures.extend(figs)

figs = plot_feature_summary_pages(model,
                                  X_train.columns,
                                  target_cols,
                                  mae_list,
                                  rmse_list,
                                  r2_list,
                                  steps,
                                  has_heatmap=False,
                                  top_n=15)
figures.extend(figs)

###############################################################################
# === Save Report ===
end_time = time.time()
elapsed_time_str = f"{int((end_time-start_time)//60)} min {int((end_time-start_time)%60)} sec"

model_params = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Number of data points in the train set": len(y_train),
    "Number of data points in the test set": len(y_test),
    "Forecast horizon (hours)": forecast_horizon,
    "Model": "LinearRegression (multi-output)",
    "Elapsed time": elapsed_time_str,
    "Number of used features": len(features),
    "Features": features,
}

errors = [
    (target_cols[i], r2_list[i], mae_list[i], rmse_list[i])
    for i in range(forecast_horizon)
]

save_report_to_pdf(f'LinearRegressionMultistep_report_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf',
                   f'{station_name} - オキシダント予測の分析 (Linear Regression)', model_params, errors, figures)
