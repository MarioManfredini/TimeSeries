# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utility import load_and_prepare_data

# === Caricamento dati ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data_all, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

# === Colonne usate e lag fino al 2 ===
cols = ['SO2(ppm)', 'NO(ppm)', 'NO2(ppm)', 'NMHC(ppmC)', 'CH4(ppmC)',
        'SPM(mg/m3)', 'PM2.5(μg/m3)', 'U', 'V', 'TEMP(℃)', 'HUM(％)', 'Ox(ppm)']
data = data_all[cols].dropna()

# === Verifica stazionarietà (ADF test) ===
def check_stationarity(series, signif=0.05):
    if series.nunique() <= 1:
        return False
    try:
        result = adfuller(series.dropna(), autolag='AIC')
        return result[1] < signif
    except:
        return False

stationary_flags = {col: check_stationarity(data[col]) for col in data.columns}
print("Stazionarietà iniziale:", stationary_flags)

# === Differenziazione se necessaria ===
data_diff = data.copy()
for col in data.columns:
    if not stationary_flags[col]:
        data_diff[col] = data[col].diff()
data_diff = data_diff.dropna()

# === STL Decomposition della sola serie Ox ===
stl = STL(data["Ox(ppm)"], period=24)
res = stl.fit()
res.plot()
plt.suptitle("STL Decomposition - Ox(ppm)")
plt.show()

# === Selezione automatica lag per VAR ===
model_selection = VAR(data_diff)
lag_order = model_selection.select_order(maxlags=48)
selected_lag = lag_order.aic
print("Lag consigliato (AIC):", selected_lag)

# === Divisione retroattiva in train/test ===
forecast_steps = 24
split_point = -forecast_steps
train_data = data_diff.iloc[:split_point]
test_data = data_diff.iloc[split_point - selected_lag:]

# === Fit del modello VAR sui dati di training ===
model = VAR(train_data)
model_fitted = model.fit(selected_lag)
print(model_fitted.summary())

# === Previsione sulle 24 ore successive (test retroattivo) ===
forecast_input = test_data.values[:selected_lag]
forecast = model_fitted.forecast(forecast_input, steps=forecast_steps)
forecast_index = data_diff.iloc[split_point:].index[:forecast_steps]
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=data_diff.columns)

# === Inversione differenza su Ox se era stata differenziata ===
if not stationary_flags["Ox(ppm)"]:
    last_real_value = data["Ox(ppm)"].iloc[split_point - 1]
    forecast_df["Ox(ppm)"] = forecast_df["Ox(ppm)"].cumsum() + last_real_value

# === Confronto predizione vs valori reali ===
true_values = data["Ox(ppm)"].loc[forecast_df.index]
predicted_values = forecast_df["Ox(ppm)"]

plt.figure(figsize=(10, 5))
plt.plot(true_values, label="Reale")
plt.plot(predicted_values, label="Predetto", linestyle="--")
plt.title("Confronto Ox(ppm): Reale vs Predetto (VAR)")
plt.legend()
plt.grid()
plt.show()

# === Metriche di valutazione ===
print("R^2:", r2_score(true_values, predicted_values))
print("MAE:", mean_absolute_error(true_values, predicted_values))
print("RMSE:", np.sqrt(mean_squared_error(true_values, predicted_values)))
