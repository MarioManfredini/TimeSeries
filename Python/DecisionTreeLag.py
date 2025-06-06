# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

target_item = 'Ox(ppm)'

###############################################################################
# === Caricamento dati ===
from utility import load_and_prepare_data
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# === Dati ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

# === Lag fino al 2 ===
lags = 2
lagged_items = ['SO2(ppm)', 'NO(ppm)', 'NO2(ppm)',
                'NMHC(ppmC)', 'CH4(ppmC)', 'SPM(mg/m3)', 'PM2.5(μg/m3)',
                'U', 'V', 'TEMP(℃)', 'HUM(％)', 'Ox(ppm)']

target_item = 'Ox(ppm)'  # La variabile target

lagged_data = pd.DataFrame()

for item in lagged_items:
    for lag in range(1, lags + 1):
        lagged_data[f"{item}_lag{lag}"] = data[item].shift(lag)

# Target corrente
lagged_data[target_item] = data[target_item]

# Rimuovi righe con NaN
lagged_data = lagged_data.dropna()

# === Feature e Target ===
X_lagged = lagged_data.drop(columns=[target_item])
y_lagged = lagged_data[target_item]

# === Divisione 70% train - 30% test ===
split_index = int(len(X_lagged) * 0.7)
X_train_lagged = X_lagged.iloc[:split_index]
X_test_lagged = X_lagged.iloc[split_index:]
y_train_lagged = y_lagged.iloc[:split_index]
y_test_lagged = y_lagged.iloc[split_index:]

###############################################################################
# === Modello: Singolo Albero Decisionale ===
#tree = DecisionTreeRegressor(max_depth=6, random_state=42)
tree = DecisionTreeRegressor(min_samples_leaf=40, random_state=42)
tree.fit(X_train_lagged, y_train_lagged)

# === Predizioni ===
y_pred_train = tree.predict(X_train_lagged)
y_pred_test = tree.predict(X_test_lagged)

# === Valutazione ===
print("\nR² Train:", r2_score(y_train_lagged, y_pred_train))
print("R² Test:", r2_score(y_test_lagged, y_pred_test))
print("\nMAE:", mean_absolute_error(y_test_lagged, y_pred_test))
print("MSE:", mean_squared_error(y_test_lagged, y_pred_test))
print("RMSE:", np.sqrt(mean_squared_error(y_test_lagged, y_pred_test)))

# === Plot ===
plt.figure(figsize=(12, 5))
plt.plot(y_test_lagged.index, y_test_lagged.values, label='Valori reali Ox', marker='o', markersize=3, linestyle='-', alpha=0.7)
plt.plot(y_test_lagged.index, y_pred_test, label='Predizioni Albero', marker='x', linestyle='--', alpha=0.7)
plt.xlabel('Data')
plt.ylabel('Ox(ppm)')
plt.title('Confronto tra valori reali e predetti (Albero Decisionale)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
