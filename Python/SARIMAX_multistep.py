# -*- coding: utf-8 -*-
"""
SARIMAX Rolling Forecast (Ox)
Created 2025/06
@author: Mario
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utility import get_station_name, load_and_prepare_data
from report import save_report_to_pdf, plot_comparison_actual_predicted, plot_feature_summary_pages
import time

start_time = time.time()

###############################################################################
# === Parameters ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'
forecast_horizon = 24

###############################################################################
# === Load and preprocess data ===
data, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

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

data_model = data.dropna(subset=features).copy()

# === Normalize features ===
scaler = MinMaxScaler()
data_model[features] = scaler.fit_transform(data_model[features])

# === Shift target for one-step prediction ===
data_model[target_item] = data_model[target_item].shift(-1)
data_model = data_model.dropna()

# === Train/Test Split ===
y = data_model[target_item]
X = data_model[features]
split_index = int(len(y) * 0.7)
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]

# === Rolling Forecast ===
mae_list, rmse_list, r2_list = [], [], []
true_vals = []
multi_step_preds = []

train_y = y_train.copy()
train_X = X_train.copy()

for step in range(forecast_horizon):
    if step >= len(y_test):
        break

    # Fit the model at each step on updated training data
    model = SARIMAX(train_y,
                    exog=train_X,
                    order=(1, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)

    # Predict one-step ahead
    x_exog = X_test.iloc[step:step+1]
    y_true = y_test.iloc[step]
    y_pred = results.predict(start=len(train_y), end=len(train_y), exog=x_exog).iloc[0]

    multi_step_preds.append(y_pred)
    true_vals.append(y_true)

    # Append the observed value to training set (rolling strategy)
    train_y = pd.concat([train_y, pd.Series([y_true], index=[x_exog.index[0]])])
    train_X = pd.concat([train_X, x_exog])

    # Compute errors
    mae_list.append(mean_absolute_error([y_true], [y_pred]))
    rmse_list.append(np.sqrt(mean_squared_error([y_true], [y_pred])))
    r2_list.append(np.nan)  # R² not defined for a single point

###############################################################################
# === Prepare PDF Report ===
figures = []

steps = [f't+{i:02}' for i in range(1, len(true_vals) + 1)]
target_cols = [f'{target_item}_t+{i:02}' for i in range(1, len(true_vals) + 1)]

# === Feature Summary Plot ===
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

# === Save Report ===
end_time = time.time()
elapsed_time_str = f"{int((end_time - start_time)//60)} min {int((end_time - start_time)%60)} sec"

model_params = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Number of data points in the train set": len(y_train),
    "Number of data points in the test set": len(y_test),
    "Forecast horizon (hours)": forecast_horizon,
    "Model": "SARIMAX (rolling forecast)",
    "Elapsed time": elapsed_time_str,
    "Number of used features": len(features),
    "Features": features,
}

errors = [
    (target_cols[i], r2_list[i], mae_list[i], rmse_list[i])
    for i in range(len(multi_step_preds))
]

save_report_to_pdf(f'SARIMAX_rolling_report_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf',
                   f'{station_name} - Analysis of Ox Forecast (SARIMAX rolling)', model_params, errors, figures)
