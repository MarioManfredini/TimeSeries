# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from utility import load_and_prepare_data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

###############################################################################
# === Parametri ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
forecast_horizon = 8  # n-step forecast

###############################################################################
# === Caricamento dati ===
data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Feature engineering ===
target_item = 'Ox(ppm)'
lagged_items = ['NO(ppm)', 'NO2(ppm)', 'U', 'V', target_item]
lags = 48

features = []
for item in lagged_items:
    for lag in range(1, lags + 1):
        data[f"{item}_lag{lag}"] = data[item].shift(lag)
        features.append(f"{item}_lag{lag}")

data[f'{target_item}_roll_mean_3'] = data[target_item].rolling(3).mean()
data['NO(ppm)_roll_mean_3'] = data['NO(ppm)'].rolling(window=3).mean()
data['NO2(ppm)_roll_mean_3'] = data['NO2(ppm)'].rolling(window=3).mean()
data['U_roll_mean_3'] = data['U'].rolling(window=3).mean()
data['V_roll_mean_3'] = data['V'].rolling(window=3).mean()

data[f'{target_item}_roll_std_6'] = data[target_item].rolling(6).std()
data['NO(ppm)_roll_std_6'] = data['NO(ppm)'].rolling(window=6).std()
data['NO2(ppm)_roll_std_6'] = data['NO2(ppm)'].rolling(window=6).std()
data['U_roll_std_6'] = data['U'].rolling(window=6).std()
data['V_roll_std_6'] = data['V'].rolling(window=6).std()

data[f'{target_item}_diff_1'] = data[target_item].diff(1)
data[f'{target_item}_diff_2'] = data[target_item].diff(2)
data[f'{target_item}_diff_3'] = data[target_item].diff(3)
data['NO(ppm)_diff_3'] = data['NO(ppm)'].diff(3)
data['NO2(ppm)_diff_3'] = data['NO2(ppm)'].diff(3)
data['U_diff_3'] = data['U'].diff(3)
data['V_diff_3'] = data['V'].diff(3)

features += [
    f'{target_item}_roll_mean_3',
    'NO(ppm)_roll_mean_3',
    'NO2(ppm)_roll_mean_3',
    'U_roll_mean_3',
    'V_roll_mean_3',
    f'{target_item}_roll_std_6',
    'NO(ppm)_roll_std_6',
    'NO2(ppm)_roll_std_6',
    'U_roll_std_6',
    'V_roll_std_6',
    f'{target_item}_diff_1',
    f'{target_item}_diff_2',
    f'{target_item}_diff_3',
    'NO(ppm)_diff_3',
    'NO2(ppm)_diff_3',
    'U_diff_3',
    'V_diff_3',
    ]

# === Creazione target multi-output ===
for i in range(forecast_horizon):
    data[f'{target_item}_t+{i+1}'] = data[target_item].shift(-i-1)

target_cols = [f'{target_item}_t+{i+1}' for i in range(forecast_horizon)]

data_model = data.dropna(subset=features + target_cols).copy()

###############################################################################
# === Normalizzazione ===
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(data_model[features])
y = scaler_y.fit_transform(data_model[target_cols])

###############################################################################
# === Split ===
split_index = int(len(X) * 0.7)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

###############################################################################
# === Modello ===
base_model = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    n_estimators=400,
    learning_rate=0.05,
    max_depth=10,
    random_state=42,
)

model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

###############################################################################
# === Previsione ===
X_test_df = pd.DataFrame(X_test, columns=features)
y_pred_scaled = model.predict(X_test_df)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

###############################################################################
# === Visualizzazione ===
plt.figure(figsize=(12, 5))
t_index = data_model.index[split_index:split_index+len(y_pred)]

print("Errore per ciascun passo di previsione (t+1, t+2, ...):")
for i, col in enumerate(target_cols):
    r2_scores = r2_score(y_true[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    print(f"{col}: R²: {r2_scores:.4f} MAE = {mae:.4f}, RMSE = {rmse:.4f}")

# Etichette per ogni passo di previsione
steps = ['t+1', 't+2', 't+3', 't+4', 't+5', 't+6', 't+7', 't+8']

plt.figure(figsize=(16, 10))
for i in range(8):
    plt.subplot(4, 2, i + 1)
    plt.plot(y_true[:300, i], label='Reale', color='black', linewidth=1)
    plt.plot(y_pred[:300, i], label='Predetto', color='red', linestyle='--', linewidth=1)
    plt.title(f'Ox(ppm) - Step {steps[i]}')
    plt.xlabel('Campioni')
    plt.ylabel('Ox(ppm)')
    plt.legend()
    plt.tight_layout()

plt.suptitle('Confronto tra valori reali e predetti', fontsize=16, y=1.02)
plt.show()