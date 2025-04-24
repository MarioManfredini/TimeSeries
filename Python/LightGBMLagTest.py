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

# === Dati ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

# === Lag ===
lagged_items = ['NO(ppm)', 'NO2(ppm)',
                'U', 'V',
                'TEMP(℃)', 'HUM(％)',
                'Ox(ppm)']

target_item = 'Ox(ppm)'

lagged_data_lag2 = pd.DataFrame()
lags = 2
for item in lagged_items:
    for lag in range(1, lags + 1):
        lagged_data_lag2[f"{item}_lag{lag}"] = data[item].shift(lag)
# Target corrente
lagged_data_lag2[target_item] = data[target_item]
# Rimuovi righe con NaN
lagged_data_lag2 = lagged_data_lag2.dropna()
# === Feature e Target ===
X_lag2 = lagged_data_lag2.drop(columns=[target_item])
y_lag2 = lagged_data_lag2[target_item]
# === Divisione 70% train - 30% test ===
split_index = int(len(X_lag2) * 0.7)
X_train_lag2 = X_lag2.iloc[:split_index]
X_test_lag2 = X_lag2.iloc[split_index:]
y_train_lag2 = y_lag2.iloc[:split_index]
y_test_lag2 = y_lag2.iloc[split_index:]

lagged_data_lag3 = pd.DataFrame()
lags = 3
for item in lagged_items:
    for lag in range(1, lags + 1):
        lagged_data_lag3[f"{item}_lag{lag}"] = data[item].shift(lag)
# Target corrente
lagged_data_lag3[target_item] = data[target_item]
# Rimuovi righe con NaN
lagged_data_lag3 = lagged_data_lag3.dropna()
# === Feature e Target ===
X_lag3 = lagged_data_lag3.drop(columns=[target_item])
y_lag3 = lagged_data_lag3[target_item]
# === Divisione 70% train - 30% test ===
split_index = int(len(X_lag3) * 0.7)
X_train_lag3 = X_lag3.iloc[:split_index]
X_test_lag3 = X_lag3.iloc[split_index:]
y_train_lag3 = y_lag3.iloc[:split_index]
y_test_lag3 = y_lag3.iloc[split_index:]

###############################################################################
import lightgbm as lgb

# === Definizione del modello ===
lgb_model = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.6,
    colsample_bytree=1.0,
    min_data_in_leaf=20,
    random_state=42
)

# === Allenamento ===
lgb_model.fit(X_train_lag2, y_train_lag2)
# === Predizione ===
y_pred_lag2 = lgb_model.predict(X_test_lag2)

# === Allenamento ===
lgb_model.fit(X_train_lag3, y_train_lag3)
# === Predizione ===
y_pred_lag3 = lgb_model.predict(X_test_lag3)



import matplotlib.pyplot as plt
import seaborn as sns

# Imposta stile
sns.set(style="whitegrid")

# Creazione figure con sottopannelli
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# --- Scatter plot: y_test vs y_pred ---
axs[0, 0].scatter(y_test_lag2, y_pred_lag2, alpha=0.6, label='Lag 2', color='royalblue')
axs[0, 0].plot([y_test_lag2.min(), y_test_lag2.max()], [y_test_lag2.min(), y_test_lag2.max()], 'r--')
axs[0, 0].set_title('Lag 2 - Reale vs Predetto')
axs[0, 0].set_xlabel('Valore Reale')
axs[0, 0].set_ylabel('Valore Predetto')

axs[1, 0].scatter(y_test_lag3, y_pred_lag3, alpha=0.6, label='Lag 3', color='darkorange')
axs[1, 0].plot([y_test_lag3.min(), y_test_lag3.max()], [y_test_lag3.min(), y_test_lag3.max()], 'r--')
axs[1, 0].set_title('Lag 3 - Reale vs Predetto')
axs[1, 0].set_xlabel('Valore Reale')
axs[1, 0].set_ylabel('Valore Predetto')

# --- Residui: y_test - y_pred ---
residuals_lag2 = y_test_lag2 - y_pred_lag2
residuals_lag3 = y_test_lag3 - y_pred_lag3

sns.histplot(residuals_lag2, kde=True, ax=axs[0, 1], color='royalblue')
axs[0, 1].set_title('Distribuzione Residui - Lag 2')
axs[0, 1].set_xlabel('Residuo')

sns.histplot(residuals_lag3, kde=True, ax=axs[1, 1], color='darkorange')
axs[1, 1].set_title('Distribuzione Residui - Lag 3')
axs[1, 1].set_xlabel('Residuo')

# --- Q-Q Plot dei residui ---
import scipy.stats as stats

stats.probplot(residuals_lag2, dist="norm", plot=axs[0, 2])
axs[0, 2].set_title('Q-Q Plot Residui - Lag 2')

stats.probplot(residuals_lag3, dist="norm", plot=axs[1, 2])
axs[1, 2].set_title('Q-Q Plot Residui - Lag 3')

plt.tight_layout()
plt.show()

plt.figure(figsize=(14,5))
plt.plot(y_test_lag2.index, y_test_lag2, label="Valori reali", color='black')
plt.plot(y_test_lag2.index, y_pred_lag2, label="Predetto - Lag 2", linestyle="--", color='royalblue')
plt.plot(y_test_lag3.index, y_pred_lag3, label="Predetto - Lag 3", linestyle="--", color='darkorange')
plt.legend()
plt.title('Confronto temporale tra reale e predetto')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Supponendo che tu abbia già:
# y_test, y_pred_lag2, y_pred_lag3

# Calcolo residui
residuals_lag2 = y_test_lag2 - y_pred_lag2
residuals_lag3 = y_test_lag3 - y_pred_lag3

# --- 1️⃣ Rolling Window Error (es. 100 samples) ---
rolling_mae_lag2 = residuals_lag2.abs().rolling(100).mean()
rolling_mae_lag3 = residuals_lag3.abs().rolling(100).mean()

plt.figure(figsize=(12,6))
plt.plot(rolling_mae_lag2, label='Lag 2 - Rolling MAE')
plt.plot(rolling_mae_lag3, label='Lag 3 - Rolling MAE', alpha=0.75)
plt.xlabel('Samples')
plt.ylabel('Rolling MAE (window=100)')
plt.title('Andamento Rolling MAE nel tempo')
plt.legend()
plt.grid()
plt.show()

# --- 2️⃣ Distribuzione Residui (KDE Plot) ---
plt.figure(figsize=(12,6))
sns.kdeplot(residuals_lag2, label='Lag 2 Residuals', fill=True, alpha=0.4)
sns.kdeplot(residuals_lag3, label='Lag 3 Residuals', fill=True, alpha=0.4)
plt.title('Distribuzione dei residui')
plt.xlabel('Errore')
plt.legend()
plt.grid()
plt.show()

# --- 3️⃣ Outlier Detection Z-Score ---
z_scores_lag2 = np.abs(stats.zscore(residuals_lag2))
z_scores_lag3 = np.abs(stats.zscore(residuals_lag3))

outliers_lag2 = np.sum(z_scores_lag2 > 3)
outliers_lag3 = np.sum(z_scores_lag3 > 3)

print("Numero di outlier (Z-score > 3):")
print(f"Lag 2: {outliers_lag2}")
print(f"Lag 3: {outliers_lag3}")

# --- 4️⃣ Outlier Detection con IQR ---
def iqr_outliers(residuals):
    q1 = np.percentile(residuals, 25)
    q3 = np.percentile(residuals, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return np.sum((residuals < lower) | (residuals > upper))

outliers_iqr_lag2 = iqr_outliers(residuals_lag2)
outliers_iqr_lag3 = iqr_outliers(residuals_lag3)

print("Numero di outlier (IQR method):")
print(f"Lag 2: {outliers_iqr_lag2}")
print(f"Lag 3: {outliers_iqr_lag3}")
