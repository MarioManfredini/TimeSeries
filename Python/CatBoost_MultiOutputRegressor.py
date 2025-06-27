# -*- coding: utf-8 -*-
"""
CatBoost multi-step forecasting for Ox(ppm)

Created 2025/06/21
@author: Mario
"""

import pandas as pd
import numpy as np
import time
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor

from utility import get_station_name, load_and_prepare_data
from report import save_report_to_pdf, plot_feature_summary_pages, plot_comparison_actual_predicted

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
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Feature Engineering ===
features = ['NO(ppm)', 'NO2(ppm)', 'U', 'V']
lagged_items = [target_item] + features

lag_features = {}
for item in lagged_items:
    for l in range(1, 24):
        col = f'{item}_lag{l}'
        lag_features[col] = df[item].shift(l)
        features.append(col)
df = pd.concat([df, pd.DataFrame(lag_features, index=df.index)], axis=1)

rolling_features = {}
for item in lagged_items:
    rolling_features[f'{item}_roll_mean_3'] = df[item].shift(1).rolling(3).mean()
    rolling_features[f'{item}_roll_std_6'] = df[item].shift(1).rolling(6).std()
    features += [f'{item}_roll_mean_3', f'{item}_roll_std_6']
df = pd.concat([df, pd.DataFrame(rolling_features, index=df.index)], axis=1)

diff_features = {
    f'{target_item}_diff_{i}': df[target_item].shift(1).diff(i) for i in range(1, 4)
}
for item in features[:4]:  # ['NO(ppm)', 'NO2(ppm)', 'U', 'V']
    diff_features[f'{item}_diff_3'] = df[item].shift(1).diff(3)
features += list(diff_features.keys())
df = pd.concat([df, pd.DataFrame(diff_features, index=df.index)], axis=1)

df['hour_sin'] = np.sin(2 * np.pi * df['時'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['時'] / 24)
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
features += ['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend']

# Multi-output target
for i in range(forecast_horizon):
    df[f'{target_item}_t+{i+1:02}'] = df[target_item].shift(-i - 1)
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
# === Model Training (CatBoost) ===
base_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    verbose=False,
    random_seed=42
)
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
r2_list = []
mae_list = []
rmse_list = []
for i, col in enumerate(target_cols):
    r2 = r2_score(y_true[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    r2_list.append(r2)
    mae_list.append(mae)
    rmse_list.append(rmse)
    print(f"{col}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

###############################################################################
# === Report ===
figures = []

steps = [f't+{i:02}' for i in range(1, forecast_horizon + 1)]
figs = plot_comparison_actual_predicted(y_true, y_pred, target_cols, steps, rows_per_page=3)
figures.extend(figs)

figs = plot_feature_summary_pages(model, X_train.columns, target_cols, mae_list, rmse_list, r2_list, steps, top_n=15)
figures.extend(figs)

end_time = time.time()
elapsed_time = f"{int((end_time - start_time)//60)} min {int((end_time - start_time)%60)} sec"

model_params = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Forecast horizon": forecast_horizon,
    "Train size": len(X_train),
    "Test size": len(X_test),
    "Model": "CatBoost",
    "Iterations": base_model.get_param('iterations'),
    "Learning rate": base_model.get_param('learning_rate'),
    "Depth": base_model.get_param('depth'),
    "Loss function": base_model.get_param('loss_function'),
    "Elapsed time": elapsed_time,
    "Number of used features": len(features),
    "Features": features,
}

errors = [
    (target_cols[i], r2_list[i], mae_list[i], rmse_list[i])
    for i in range(forecast_horizon)
]

save_report_to_pdf(f'CatBoost_report_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf',
                   f'{station_name} - オキシダント予測の分析 (CatBoost)', model_params, errors, figures)
