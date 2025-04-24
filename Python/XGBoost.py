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
lagged_items = ['SO2(ppm)', 'NO(ppm)', 'NO2(ppm)',
               'NMHC(ppmC)', 'CH4(ppmC)', 'SPM(mg/m3)', 'PM2.5(μg/m3)',
                'U', 'V', 'TEMP(℃)', 'HUM(％)', 'Ox(ppm)']
#lagged_items = ['NO(ppm)', 'Ox(ppm)']

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
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Definizione del modello base
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Definizione della griglia di parametri
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Configurazione del GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='r2',
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Esecuzione del Grid Search
grid_search.fit(X_train_lagged, y_train_lagged)

# Migliori parametri trovati
print("Migliori parametri trovati:")
print(grid_search.best_params_)

# Valutazione sul set di test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_lagged)

# Calcolo delle metriche
r2 = r2_score(y_test_lagged, y_pred)
mae = mean_absolute_error(y_test_lagged, y_pred)
mse = mean_squared_error(y_test_lagged, y_pred)
rmse = np.sqrt(mse)

print(f"R²: {r2:.5f}")
print(f"MAE: {mae:.5f}")
print(f"MSE: {mse:.5f}")
print(f"RMSE: {rmse:.5f}")

# Confronto Visuale Predizioni vs. Osservazioni
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 2)
plt.plot(y_test_lagged.values, label='Valori Reali', color='black')
plt.plot(y_pred, label='XGBoost (con lag)', color='green', linestyle='dashed')
plt.title(f'Regressione con lag 1-2\nR²: {r2:.5f}')
plt.xlabel('Campioni')
plt.ylabel(target_item)
plt.legend()
plt.grid(True)
plt.show()

# Errore Residuo
residuals = y_test_lagged.values - y_pred

plt.figure(figsize=(8,4))
plt.scatter(range(len(residuals)), residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Campioni')
plt.ylabel('Errore (Osservato - Predetto)')
plt.title('Distribuzione degli Errori Residui')
plt.grid(True)
plt.show()

# Importanza delle Variabili
import pandas as pd
importance = pd.Series(best_model.feature_importances_, index=X_train_lagged.columns)
importance_sorted = importance.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importance_sorted.plot(kind='barh', color='skyblue')
plt.xlabel('Importanza')
plt.title('Importanza delle Variabili - XGBoost')
plt.grid(True, axis='x')
plt.show()

# E stampa anche i valori in console
print("\nFeature Importance dettagliata:")
for feature, score in importance_sorted.sort_values(ascending=False).items():
    print(f"{feature}: {score:.5f}")
