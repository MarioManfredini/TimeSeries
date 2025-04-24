# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

###############################################################################
# === Caricamento dati ===
from utility import load_and_prepare_data

data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# Creazione lag fino al 2
lags = 2
lagged_items = ['SO2(ppm)', 'NO(ppm)', 'NO2(ppm)',
           'NMHC(ppmC)', 'CH4(ppmC)', 'SPM(mg/m3)', 'PM2.5(μg/m3)',
           'U', 'V', 'TEMP(℃)', 'HUM(％)', 'Ox(ppm)']
#lagged_items = ['NO(ppm)', 'Ox(ppm)']

target_item = 'Ox(ppm)'

lagged_data = pd.DataFrame()

for item in lagged_items:
    for lag in range(1, lags + 1):
        lagged_data[f"{item}_lag{lag}"] = data[item].shift(lag)

# Aggiungi la variabile target attuale
lagged_data[target_item] = data[target_item]

# Rimuovi i NaN
lagged_data = lagged_data.dropna()

# Feature / Target
X_lagged = lagged_data.drop(columns=[target_item])
y_lagged = lagged_data[target_item]

# Divisione senza shuffle
split_index = int(len(X_lagged) * 0.7)
X_train_lagged = X_lagged.iloc[:split_index]
X_test_lagged = X_lagged.iloc[split_index:]
y_train_lagged = y_lagged.iloc[:split_index]
y_test_lagged = y_lagged.iloc[split_index:]

###############################################################################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Modello
model_lagged = LinearRegression()
model_lagged.fit(X_train_lagged, y_train_lagged)

# Predizione
y_pred_lagged = model_lagged.predict(X_test_lagged)

# Valutazione
r2_lagged = r2_score(y_test_lagged, y_pred_lagged)
print(f"\nR² con variabili ritardate: {r2_lagged:.5f}")

# Imposta dimensione figura
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 2)
plt.plot(y_test_lagged.values, label='Valori Reali', color='black')
plt.plot(y_pred_lagged, label='Predetti (con lag)', color='green', linestyle='dashed')
plt.title(f'Regressione con lag 1-2\nR²: {r2_lagged:.5f}')
plt.xlabel('Campioni')
plt.ylabel(target_item)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

residuals_lagged = y_test_lagged.values - y_pred_lagged

plt.figure(figsize=(12, 6))

# Istogramma per modello con lag
plt.subplot(1, 2, 2)
plt.hist(residuals_lagged, bins=30, color='green', alpha=0.7, edgecolor='black')
plt.title('Errori Residui - Modello Con Lag')
plt.xlabel('Errore')
plt.ylabel('Frequenza')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calcola MAE, MSE, RMSE
mae = mean_absolute_error(y_test_lagged, y_pred_lagged)
mse = mean_squared_error(y_test_lagged, y_pred_lagged)
rmse = np.sqrt(mse)

# Stampa i risultati
print(f"\nmedia misurazioni: {np.average(data[target_item]):.5f}")
print(f"MAE (Errore Assoluto Medio) - Modello Con Lag: {mae:.5f}")
print(f"MSE (Errore Quadratico Medio) - Modello Con Lag: {mse:.5f}")
print(f"RMSE (Radice Errore Quadratico Medio) - Modello Con Lag: {rmse:.5f}")
