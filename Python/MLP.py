# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from utility import load_and_prepare_data

# === Caricamento dati ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data_all, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

# === Colonne usate ===
cols = ['SO2(ppm)', 'NO(ppm)', 'NO2(ppm)', 'NMHC(ppmC)', 'CH4(ppmC)',
        'SPM(mg/m3)', 'PM2.5(μg/m3)', 'U', 'V', 'TEMP(℃)', 'HUM(％)', 'Ox(ppm)']
target_item = 'Ox(ppm)'
data = data_all[cols].dropna()

# === Creazione delle feature lag ===
"""
def create_lag_features(df, target_col, lags=2):
    df_lagged = pd.DataFrame(index=df.index)
    for col in df.columns:
        for lag in range(1, lags + 1):
            df_lagged[f"{col}_lag{lag}"] = df[col].shift(lag)
    df_lagged[target_col] = df[target_col]
    return df_lagged.dropna()

df_lagged = create_lag_features(data, target_col=target_item, lags=2)

# === Divisione in X (input) e y (target) ===
X = df_lagged.drop(columns=[target_item])
y = df_lagged[target_item]
"""

df_lagged = data[[target_item]].copy()
df_lagged["lag1"] = df_lagged[target_item].shift(1)
df_lagged["lag2"] = df_lagged[target_item].shift(2)
df_lagged = df_lagged.dropna()

X = df_lagged[["lag1", "lag2"]]
y = df_lagged[target_item]



# === Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# === Standardizzazione ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === MLP ===
mlp = MLPRegressor(hidden_layer_sizes=(100, 80), activation='relu', max_iter=20, random_state=42)
mlp.fit(X_train_scaled, y_train)

# === Predizioni ===
y_pred = mlp.predict(X_test_scaled)

# === Metriche ===
print("R^2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# === Grafico ===
plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label="Reale")
plt.plot(y_pred, label="Predetto", linestyle="--")
plt.title("Previsione Ox(ppm) con MLP")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
