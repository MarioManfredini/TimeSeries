# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

###############################################################################
# === Caricamento dati ===
from utility import load_and_prepare_data

data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Lag e feature derivate ===

#lagged_items = ['NOx(ppm)', 'U', 'V', 'TEMP(℃)', 'HUM(％)', 'Ox(ppm)']
lagged_items = ['SO2(ppm)', 'NO(ppm)', 'NO2(ppm)', 'U', 'V', 'Ox(ppm)']
target_item = 'Ox(ppm)'
lags = 4

lagged_data = pd.DataFrame(index=data.index)  # Assicura stesso indice

# LAG
for item in lagged_items:
    for lag in range(1, lags + 1):
        lagged_data[f"{item}_lag{lag}"] = data[item].shift(lag)

# Target
lagged_data[target_item] = data[target_item]

lagged_data[f'{target_item}_lag24'] = data[target_item].shift(24)

# === Feature derivate ===

# Media mobile
lagged_data[f'{target_item}_roll_mean_3'] = data[target_item].rolling(window=3).mean()
lagged_data['NO(ppm)_roll_mean_3'] = data['NO(ppm)'].rolling(window=3).mean()
lagged_data['NO2(ppm)_roll_mean_3'] = data['NO2(ppm)'].rolling(window=3).mean()
#lagged_data['NO(ppm)_roll_mean_24'] = data['NO(ppm)'].rolling(window=24).mean()
#lagged_data['NO2(ppm)_roll_mean_24'] = data['NO2(ppm)'].rolling(window=24).mean()
lagged_data['U_roll_mean_3'] = data['U'].rolling(window=3).mean()
lagged_data['V_roll_mean_3'] = data['V'].rolling(window=3).mean()

# Deviazione standard
lagged_data[f'{target_item}_roll_std_6'] = data[target_item].rolling(window=6).std()
#lagged_data['TEMP_roll_std_6'] = data['TEMP(℃)'].rolling(window=6).std()
lagged_data['NO(ppm)_roll_std_6'] = data['NO(ppm)'].rolling(window=6).std()
lagged_data['NO2(ppm)_roll_std_6'] = data['NO2(ppm)'].rolling(window=6).std()
lagged_data['U_roll_std_6'] = data['U'].rolling(window=6).std()
lagged_data['V_roll_std_6'] = data['V'].rolling(window=6).std()

# Differenze
lagged_data[f'{target_item}_diff_1'] = data[target_item].diff(1)
lagged_data[f'{target_item}_diff_2'] = data[target_item].diff(2)
lagged_data[f'{target_item}_diff_3'] = data[target_item].diff(3)
#lagged_data['TEMP_diff_3'] = data['TEMP(℃)'].diff(3)
lagged_data['NO(ppm)_diff_3'] = data['NO(ppm)'].diff(3)
lagged_data['NO2(ppm)_diff_3'] = data['NO2(ppm)'].diff(3)
lagged_data['U_diff_3'] = data['U'].diff(3)
lagged_data['V_diff_3'] = data['V'].diff(3)

# === Feature temporali ===
lagged_data['hour'] = data.index.hour
#lagged_data['is_night'] = lagged_data['hour'].apply(lambda x: 1 if x < 6 else 0)
lagged_data['weekday'] = data.index.weekday
#lagged_data['is_weekend'] = lagged_data['weekday'].apply(lambda x: 1 if x >= 5 else 0)

# === Pulizia finale ===
lagged_data = lagged_data.dropna()

###############################################################################
# === Split dati ===

X_lagged = lagged_data.drop(columns=[target_item])
y_lagged = lagged_data[target_item]

# === TimeSeriesSplit ===
tscv = TimeSeriesSplit(n_splits=5)
r2_scores = []
mae_scores = []
rmse_scores = []
importance_df = []

for train_index, test_index in tscv.split(X_lagged):
    X_train, X_test = X_lagged.iloc[train_index], X_lagged.iloc[test_index]
    y_train, y_test = y_lagged.iloc[train_index], y_lagged.iloc[test_index]

    model = lgb.LGBMRegressor(
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
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2_scores.append(r2_score(y_test, y_pred))
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    importance_df.append(pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_ / sum(model.feature_importances_)
        }).sort_values(by='Importance', ascending=False))

# === Visualizzazione delle prestazioni nell'ultimo fold ===
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Valori Reali')
plt.plot(y_pred, label='Valori Predetti')
plt.title('Confronto tra valori reali e predetti (ultimo fold)')
plt.xlabel('Index')
plt.ylabel('Ox(ppm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Risultati medi ===
print(f"\nR² medio: {np.mean(r2_scores):.5f}")
print(f"MAE medio: {np.mean(mae_scores):.5f}")
print(f"RMSE medio: {np.mean(rmse_scores):.5f}")

for index, importance in enumerate(importance_df):
    print(f"\nSplit {index+1}:")
    print(f"R²: {r2_scores[index]:.5f}")
    print(f"MAE: {mae_scores[index]:.5f}")
    print(f"RMSE: {rmse_scores[index]:.5f}")
    print("------- Feature Importance dettagliata:")
    # === Grafico ===
    plt.figure(figsize=(10,6))
    plt.barh(importance['Feature'], importance['Importance'], color='skyblue')
    plt.xlabel("Importanza Normalizzata")
    plt.title(f"Feature Importance Split {index+1} - LightGBM")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    for index, row in importance.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.5f}")
