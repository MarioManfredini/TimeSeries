# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import pandas as pd
import numpy as np

###############################################################################
# === Caricamento dati ===
from utility import load_and_prepare_data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# === Dati ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

# === Lag fino al 2 ===
lags = 2
#lagged_items = ['SO2(ppm)', 'NO(ppm)', 'NO2(ppm)',
#                'NMHC(ppmC)', 'CH4(ppmC)', 'SPM(mg/m3)', 'PM2.5(μg/m3)',
#                'U', 'V', 'TEMP(℃)', 'HUM(％)', 'Ox(ppm)']
lagged_items = ['NO(ppm)', 'Ox(ppm)']

target_item = 'Ox(ppm)'

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
from sklearn.ensemble import RandomForestRegressor

# === Creazione e addestramento Random Forest ===
rf = RandomForestRegressor(
    n_estimators=100,       # numero di alberi nella foresta
    min_samples_leaf=40,    # stesso valore usato per l'albero singolo
    random_state=42,
    n_jobs=-1               # usa tutti i core disponibili
)

rf.fit(X_train_lagged, y_train_lagged)

# === Predizioni ===
y_pred_train_rf = rf.predict(X_train_lagged)
y_pred_test_rf = rf.predict(X_test_lagged)

# === Valutazione ===
r2_train_rf = r2_score(y_train_lagged, y_pred_train_rf)
r2_test_rf = r2_score(y_test_lagged, y_pred_test_rf)
mae_rf = mean_absolute_error(y_test_lagged, y_pred_test_rf)
mse_rf = mean_squared_error(y_test_lagged, y_pred_test_rf)
rmse_rf = np.sqrt(mse_rf)

print(f"\nR² Train: {r2_train_rf}")
print(f"R² Test: {r2_test_rf}")
print(f"\nMAE: {mae_rf}")
print(f"MSE: {mse_rf}")
print(f"RMSE: {rmse_rf}")

importances = rf.feature_importances_
features = X_train_lagged.columns
sorted_indices = np.argsort(importances)[::-1]

print("\nFeature Importance")
import pandas as pd
# Dizionario con feature e importanza
importance_dict = dict(zip(X_train_lagged.columns, importances))
# Ordinato dal più rilevante
importance_sorted = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
# Stampa
for feature, importance in importance_sorted.items():
    print(f"{feature}: {importance:.6f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.title("Feature Importance - Random Forest")
plt.bar(range(len(importances)), importances[sorted_indices], align="center")
plt.xticks(range(len(importances)), [features[i] for i in sorted_indices], rotation=90)
plt.show()
