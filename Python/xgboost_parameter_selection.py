# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import mode, skew, kurtosis

from utility import load_and_prepare_data, get_station_name
from report import save_report_to_pdf

###############################################################################
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Feature Engineering ===
features = ['NO(ppm)', 'NO2(ppm)', 'U', 'V']
lagged_items = [target_item, 'NO(ppm)', 'NO2(ppm)', 'U', 'V']

lag_features = {}
for item in lagged_items:
    for l in range(1, 24):
        col_name = f'{item}_lag{l}'
        lag_features[col_name] = df[item].shift(l)
        features.append(col_name)

df = pd.concat([df, pd.DataFrame(lag_features, index=df.index)], axis=1)

rolling_features = {}
for item in lagged_items:
    col_mean = f'{item}_roll_mean_3'
    col_std = f'{item}_roll_std_6'
    rolling_features[col_mean] = df[item].shift(1).rolling(3).mean()
    rolling_features[col_std] = df[item].shift(1).rolling(6).std()
    features += [col_mean, col_std]

df = pd.concat([df, pd.DataFrame(rolling_features, index=df.index)], axis=1)

diff_features = {}
diff_features[f'{target_item}_diff_1'] = df[target_item].shift(1).diff(1)
diff_features[f'{target_item}_diff_2'] = df[target_item].shift(1).diff(2)
diff_features[f'{target_item}_diff_3'] = df[target_item].shift(1).diff(3)
features += list(diff_features.keys())

for item in ['NO(ppm)', 'NO2(ppm)', 'U', 'V']:
    col_name = f'{item}_diff_3'
    diff_features[col_name] = df[item].shift(1).diff(3)
    features.append(col_name)

df = pd.concat([df, pd.DataFrame(diff_features, index=df.index)], axis=1)

df["hour_sin"] = np.sin(2 * np.pi * df["時"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["時"] / 24)
df["dayofweek"] = df.index.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
features += ['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend']

target_col = [target_item]

data_model = df.dropna(subset=features + target_col).copy()

###############################################################################
# === Normalization ===
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_array = scaler_X.fit_transform(data_model[features])
y_array = scaler_y.fit_transform(data_model[target_col])

X = pd.DataFrame(X_array, columns=features)
y = pd.DataFrame(y_array, columns=target_col)

###############################################################################
# === Train/Test Split ===
split_index = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

###############################################################################
# Basic model definition
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# parameters grid definition
param_grid = {
    'n_estimators': [900, 1000, 1100],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.04, 0.045, 0.05],
}

# GridSearchCV configuration
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='r2',
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Grid Search execution
grid_search.fit(X_train, y_train)

# Best parameters found
print("Best parameters found:")
print(grid_search.best_params_)

# Evaluation on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

###############################################################################
# Calculating metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R²: {r2:.5f}")
print(f"MAE: {mae:.5f}")
print(f"MSE: {mse:.5f}")
print(f"RMSE: {rmse:.5f}")

# Residual Errors
residuals = (y_test.values.flatten() - y_pred.flatten())

# Compute statistics
res_mean = np.mean(residuals)
res_median = np.median(residuals)
res_mode = float(mode(residuals, nan_policy='omit').mode)

# Ljung-Box residuals autocorrelation test
lb_test = acorr_ljungbox(residuals, lags=[1], return_df=True)
prob_Q = lb_test["lb_pvalue"].iloc[0] if not lb_test.empty else np.nan

# residuals skewness
skewness = skew(residuals)
# residuals kurtosis
kurt = kurtosis(residuals, fisher=False)  # False → kurtosis "classic" (3 = normal)

###############################################################################
# === Save Figures for Report ===
figures = []

# Predicted vs Real
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(y_test.values[-720:], label='Real values', color='gray')
ax1.plot(y_pred[-720:], label='XGBoost (with lag)', color='lightgray', linestyle='dashed')
ax1.set_title(f'Regression with lag\nR²: {r2:.5f}')
ax1.set_xlabel('Samples')
ax1.set_ylabel(target_item)
ax1.legend()
ax1.grid(True)
fig1.tight_layout()
figures.append(fig1)

# Residual Errors
residuals = (y_test.values.flatten() - y_pred.flatten())
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.scatter(range(len(residuals)), residuals, cmap='Greys', alpha=0.5, s=10, color='gray')
ax2.axhline(0, color='gray', linestyle='--')
ax2.set_xlabel('Samples')
ax2.set_ylabel('Error (Real - Predicted)')
ax2.set_title('Distribution of Residual Errors')
ax2.grid(True)
fig2.tight_layout()
figures.append(fig2)

# Histogram of residuals – additional diagnostic plot
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.hist(residuals, bins=50, color='lightgray', edgecolor='gray', alpha=0.75)
ax3.axvline(res_mean, color='black', linestyle='--', linewidth=1, label=f'Mean = {res_mean:.5f}')
ax3.axvline(res_median, color='black', linestyle='-.', linewidth=1, label=f'Median = {res_median:.5f}')
ax3.axvline(res_mode, color='black', linestyle=':', linewidth=1, label=f'Mode = {res_mode:.5f}')
ax3.set_title('Histogram of Residuals – Distribution & Central Tendency')
ax3.set_xlabel('Residual value')
ax3.set_ylabel('Frequency')
ax3.legend()
fig3.tight_layout()
figures.append(fig3)

# Feature Importance
importance = pd.Series(best_model.feature_importances_, index=X_train.columns)
importance_sorted = importance.sort_values(ascending=True).tail(20)

fig4, ax4 = plt.subplots(figsize=(8, 5))
bars = ax4.barh(importance_sorted.index, importance_sorted.values, color='gray')
ax4.set_title('Top 20 Feature Importances (XGBoost)')
ax4.set_xlabel('Importance')
ax4.set_ylabel('Features')
ax4.grid(True, axis='x', linestyle='--', alpha=0.7)

for bar in bars:
    width = bar.get_width()
    ax4.text(
        width + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.3f}",
        va='center', ha='left', fontsize=10
    )
fig4.tight_layout()
figures.append(fig4)

plt.show()

###############################################################################
# === Save Report to PDF ===
report_file = f'xgboost_parameter_selection_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf'
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

params_info = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Number of data points in the train set": len(y_train),
    "Number of data points in the test set": len(y_test),
    "Number of features used": len(features),
    "Model": "XGBoost",
    "Objective": xgb_model.get_params()['objective'],
    "Booster": xgb_model.get_params().get('booster', 'gbtree'),
}
# --- Parameter Grid (tested) ---
for key, values in param_grid.items():
    params_info[f"Parameter Grid (tested) {key}"] = str(values)
# --- Best Parameters (found) ---
for key, value in grid_search.best_params_.items():
    params_info[f"Best Parameter (found) {key}"] = str(value)
params_info["Predictions mean"] = np.mean(y_pred)
params_info["Predictions std"] = np.std(y_pred)
params_info["Real mean"] = np.mean(y_test.values)
params_info["Real std"] = np.std(y_test.values)
params_info["Ljung-Box residuals autocorrelation, Prob(Q)"] = prob_Q
params_info["Residuals skew"] = skewness
params_info["Residuals kurtosis"] = kurt

errors = [
    (target_item, r2, mae, rmse)
]

save_report_to_pdf(
    filename=report_path,
    title=f"XGBoost parameter selection Report - {station_name}",
    params=params_info,
    features=features,
    errors=errors,
    figures=figures
)

print(f"\n✅ Report saved as {report_path}")

