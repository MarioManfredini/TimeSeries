# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

###############################################################################
# === Caricamento dati ===
from utility import load_and_prepare_data

data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Lag e feature derivate ===

lagged_items = ['NO(ppm)', 'NO2(ppm)', 'U', 'V', 'Ox(ppm)']
target_item = 'Ox(ppm)'
lags = 2

lagged_data = pd.DataFrame(index=data.index)  # Assicura stesso indice

# LAG
for item in lagged_items:
    for lag in range(1, lags + 1):
        lagged_data[f"{item}_lag{lag}"] = data[item].shift(lag)

# Media mobile
lagged_data[f'{target_item}_roll_mean_3'] = data[target_item].rolling(window=3).mean()
lagged_data['NO(ppm)_roll_mean_3'] = data['NO(ppm)'].rolling(window=3).mean()
lagged_data['NO2(ppm)_roll_mean_3'] = data['NO2(ppm)'].rolling(window=3).mean()
lagged_data['U_roll_mean_3'] = data['U'].rolling(window=3).mean()
lagged_data['V_roll_mean_3'] = data['V'].rolling(window=3).mean()

# Deviazione standard
lagged_data[f'{target_item}_roll_std_6'] = data[target_item].rolling(window=6).std()
lagged_data['NO(ppm)_roll_std_6'] = data['NO(ppm)'].rolling(window=6).std()
lagged_data['NO2(ppm)_roll_std_6'] = data['NO2(ppm)'].rolling(window=6).std()
lagged_data['U_roll_std_6'] = data['U'].rolling(window=6).std()
lagged_data['V_roll_std_6'] = data['V'].rolling(window=6).std()

# Differenze
lagged_data[f'{target_item}_diff_1'] = data[target_item].diff(1)
lagged_data[f'{target_item}_diff_2'] = data[target_item].diff(2)
lagged_data[f'{target_item}_diff_3'] = data[target_item].diff(3)
lagged_data['NO(ppm)_diff_3'] = data['NO(ppm)'].diff(3)
lagged_data['NO2(ppm)_diff_3'] = data['NO2(ppm)'].diff(3)
lagged_data['U_diff_3'] = data['U'].diff(3)
lagged_data['V_diff_3'] = data['V'].diff(3)

for col in lagged_items:
    lagged_data[col] = data[col]

# === Pulizia finale ===
lagged_data = lagged_data.dropna()

###############################################################################
# === Normalizzazione ===

features = lagged_data.drop(columns=[target_item])
target = lagged_data[[target_item]]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(target)

X_lagged = pd.DataFrame(X_scaled, index=features.index, columns=features.columns)
y_lagged = pd.Series(y_scaled.flatten(), index=target.index)

print(f"Varianza del target (dopo scaling): {np.var(y_lagged):.6f}")
print(f"Valori min/max target (scaled): {np.min(y_lagged):.4f} ~ {np.max(y_lagged):.4f}")

plt.plot(y_lagged[-200:], label='Target (scaled)')
plt.title('Target normalizzato')
plt.grid(True)
plt.legend()
plt.show()

split_index = int(len(X_lagged) * 0.7)
X_train_lagged = X_lagged.iloc[:split_index]
X_test_lagged = X_lagged.iloc[split_index:]
y_train_lagged = y_lagged.iloc[:split_index]
y_test_lagged = y_lagged.iloc[split_index:]

###############################################################################
# === Forecasting ricorsivo per le prossime 24 ore ===

forecast_horizon = 6
last_known = lagged_data.copy()  # Dati originali (non scalati) per costruire i nuovi input
future_predictions = []

# Modello addestrato su dati scalati
model_final = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=10,
    min_child_samples=10,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
#model_final.fit(X_lagged, y_lagged)
model_final.fit(
    X_train_lagged, y_train_lagged,
    eval_set=[(X_test_lagged, y_test_lagged)],
    eval_metric='rmse',
)

importances = model_final.feature_importances_
features_names = X_lagged.columns
for name, imp in sorted(zip(features_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {imp}")

for step in range(forecast_horizon):
    last_row = last_known.iloc[-1:]
    new_index = last_row.index + pd.Timedelta(hours=1)

    new_row = {}
    for item in lagged_items:
        for lag in range(1, lags + 1):
            key = f"{item}_lag{lag}"
            if lag == 1:
                new_row[key] = last_row[item].values[0]
            else:
                new_row[key] = last_known.iloc[-lag][item]

    window_3 = last_known[target_item].iloc[-2:].tolist() + [last_row[target_item].values[0]]
    new_row[f'{target_item}_roll_mean_3'] = np.mean(window_3)
    new_row['NO(ppm)_roll_mean_3'] = np.mean(last_known['NO(ppm)'].iloc[-2:].tolist() + [last_row['NO(ppm)'].values[0]])
    new_row['NO2(ppm)_roll_mean_3'] = np.mean(last_known['NO2(ppm)'].iloc[-2:].tolist() + [last_row['NO2(ppm)'].values[0]])
    new_row['U_roll_mean_3'] = np.mean(last_known['U'].iloc[-2:].tolist() + [last_row['U'].values[0]])
    new_row['V_roll_mean_3'] = np.mean(last_known['V'].iloc[-2:].tolist() + [last_row['V'].values[0]])

    window_6 = last_known[target_item].iloc[-5:].tolist() + [last_row[target_item].values[0]]
    new_row[f'{target_item}_roll_std_6'] = np.std(window_6)
    new_row['NO(ppm)_roll_std_6'] = np.std(last_known['NO(ppm)'].iloc[-5:].tolist() + [last_row['NO(ppm)'].values[0]])
    new_row['NO2(ppm)_roll_std_6'] = np.std(last_known['NO2(ppm)'].iloc[-5:].tolist() + [last_row['NO2(ppm)'].values[0]])
    new_row['U_roll_std_6'] = np.std(last_known['U'].iloc[-5:].tolist() + [last_row['U'].values[0]])
    new_row['V_roll_std_6'] = np.std(last_known['V'].iloc[-5:].tolist() + [last_row['V'].values[0]])

    new_row[f'{target_item}_diff_1'] = last_row[target_item].values[0] - last_known[target_item].iloc[-1]
    new_row[f'{target_item}_diff_2'] = last_row[target_item].values[0] - last_known[target_item].iloc[-2]
    new_row[f'{target_item}_diff_3'] = last_row[target_item].values[0] - last_known[target_item].iloc[-3]
    new_row['NO(ppm)_diff_3'] = last_row['NO(ppm)'].values[0] - last_known['NO(ppm)'].iloc[-3]
    new_row['NO2(ppm)_diff_3'] = last_row['NO2(ppm)'].values[0] - last_known['NO2(ppm)'].iloc[-3]
    new_row['U_diff_3'] = last_row['U'].values[0] - last_known['U'].iloc[-3]
    new_row['V_diff_3'] = last_row['V'].values[0] - last_known['V'].iloc[-3]

    for col in ['NO(ppm)', 'NO2(ppm)', 'U', 'V']:
        new_row[col] = last_row[col].values[0]

    X_next = pd.DataFrame([new_row])
    X_next_scaled = pd.DataFrame(scaler_X.transform(X_next), columns=X_lagged.columns)
    y_next_scaled = model_final.predict(X_next_scaled)[0]

    y_next = scaler_y.inverse_transform([[y_next_scaled]])[0][0]

    print(f"Previsione scalata: {y_next_scaled:.4f}, dopo inversa: {y_next:.4f}")
    future_predictions.append(y_next)

    next_record = last_row.copy()
    next_record.index = new_index
    next_record[target_item] = y_next
    last_known = pd.concat([last_known, next_record])

###############################################################################
# === Visualizzazione ===

plt.figure(figsize=(10, 5))
plt.plot(lagged_data.index[-48:], lagged_data[target_item].iloc[-48:], label='Storico (48h)', color='blue')
future_index = pd.date_range(start=lagged_data.index[-1] + pd.Timedelta(hours=1), periods=forecast_horizon, freq='h')
plt.plot(future_index, future_predictions, label='Previsione (24h)', color='red', marker='o')
plt.title('Previsione Ox(ppm) - LightGBM Forecasting Ricorsivo con Normalizzazione')
plt.xlabel('Data')
plt.ylabel('Ox(ppm)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
