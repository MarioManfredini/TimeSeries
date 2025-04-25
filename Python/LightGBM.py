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

###############################################################################
# === Caricamento dati ===
from utility import load_and_prepare_data

data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Lag e feature derivate ===

lagged_items = ['NOx(ppm)', 'U', 'V', 'TEMP(℃)', 'HUM(％)', 'Ox(ppm)']
target_item = 'Ox(ppm)'
lags = 2

lagged_data = pd.DataFrame(index=data.index)  # Assicura stesso indice

# LAG
for item in lagged_items:
    for lag in range(1, lags + 1):
        lagged_data[f"{item}_lag{lag}"] = data[item].shift(lag)

# Target
lagged_data[target_item] = data[target_item]

# === Feature derivate ===

# Media mobile
lagged_data[f'{target_item}_roll_mean_3'] = data[target_item].rolling(window=3).mean()

# Deviazione standard
lagged_data['TEMP_roll_std_6'] = data['TEMP(℃)'].rolling(window=6).std()

# Differenze
lagged_data[f'{target_item}_diff_1'] = data[target_item].diff(1)
lagged_data['TEMP_diff_3'] = data['TEMP(℃)'].diff(3)

# === Feature temporali ===
lagged_data['hour'] = data.index.hour
lagged_data['is_night'] = lagged_data['hour'].apply(lambda x: 1 if x < 6 else 0)
lagged_data['weekday'] = data.index.weekday
lagged_data['is_weekend'] = lagged_data['weekday'].apply(lambda x: 1 if x >= 5 else 0)

# === Pulizia finale ===
lagged_data = lagged_data.dropna()

###############################################################################
# === Split dati ===

X_lagged = lagged_data.drop(columns=[target_item])
y_lagged = lagged_data[target_item]

split_index = int(len(X_lagged) * 0.7)
X_train_lagged = X_lagged.iloc[:split_index]
X_test_lagged = X_lagged.iloc[split_index:]
y_train_lagged = y_lagged.iloc[:split_index]
y_test_lagged = y_lagged.iloc[split_index:]

###############################################################################
"""
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

# Definisci il modello base
lgb_model = lgb.LGBMRegressor(objective='regression', random_state=42)

# Griglia dei parametri
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_data_in_leaf': [5, 10, 20]
}

# GridSearch con 3-fold cross-validation
grid_search = GridSearchCV(
    estimator=lgb_model,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    verbose=2,
    n_jobs=-1
)

# Esegui il tuning
grid_search.fit(X_train_lagged, y_train_lagged)

# Risultati
print("Migliori parametri trovati:")
print(grid_search.best_params_)

# Valutazione sul test set
best_lgb = grid_search.best_estimator_
y_pred = best_lgb.predict(X_test_lagged)

print(f"R²: {r2_score(y_test_lagged, y_pred):.5f}")
print(f"MAE: {mean_absolute_error(y_test_lagged, y_pred):.5f}")
print(f"MSE: {mean_squared_error(y_test_lagged, y_pred):.5f}")
print(f"RMSE: {mean_squared_error(y_test_lagged, y_pred)**0.5:.5f}")

"""
###############################################################################
# === Modello LightGBM ===

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

lgb_model.fit(X_train_lagged, y_train_lagged)
y_pred = lgb_model.predict(X_test_lagged)

r2 = r2_score(y_test_lagged, y_pred)
mae = mean_absolute_error(y_test_lagged, y_pred)
mse = mean_squared_error(y_test_lagged, y_pred)
rmse = np.sqrt(mse)

print(f'R²: {r2:.5f}')
print(f'MAE: {mae:.5f}')
print(f'MSE: {mse:.5f}')
print(f'RMSE: {rmse:.5f}')

###############################################################################
# === Importanza delle feature ===

importance_df = pd.DataFrame({
    'Feature': X_train_lagged.columns,
    'Importance': lgb_model.feature_importances_ / sum(lgb_model.feature_importances_)
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance dettagliata:")
for index, row in importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.5f}")

# === Grafico opzionale ===
plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel("Importanza Normalizzata")
plt.title("Feature Importance - LightGBM")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
