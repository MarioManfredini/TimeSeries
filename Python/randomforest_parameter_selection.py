# -*- coding: utf-8 -*-
"""
Created 2025/11/09
@author: Mario
Random Forest parameter selection for Ox(ppm)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.diagnostic import acorr_ljungbox

from utility import load_and_prepare_data, get_station_name
from report import save_report_to_pdf

###############################################################################
# === Settings ===
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

# Lag features
lag_features = {}
for item in lagged_items:
    for l in range(1, 24):
        col_name = f'{item}_lag{l}'
        lag_features[col_name] = df[item].shift(l)
        features.append(col_name)

df = pd.concat([df, pd.DataFrame(lag_features, index=df.index)], axis=1)

# Rolling features
rolling_features = {}
for item in lagged_items:
    col_mean = f'{item}_roll_mean_3'
    col_std = f'{item}_roll_std_6'
    rolling_features[col_mean] = df[item].shift(1).rolling(3).mean()
    rolling_features[col_std] = df[item].shift(1).rolling(6).std()
    features += [col_mean, col_std]

df = pd.concat([df, pd.DataFrame(rolling_features, index=df.index)], axis=1)

# Difference features
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
# === Train/Test Split (80/20) ===
n_samples = len(data_model)
train_size = int(0.8 * n_samples)

X_train_raw = data_model[features].iloc[:train_size]
y_train_raw = data_model[target_col].iloc[:train_size]
X_val_raw = data_model[features].iloc[train_size:]
y_val_raw = data_model[target_col].iloc[train_size:]

###############################################################################
# === Normalization ===
scaler_X = MinMaxScaler().fit(X_train_raw)
scaler_y = MinMaxScaler().fit(y_train_raw)

X_train = pd.DataFrame(scaler_X.transform(X_train_raw), columns=features)
X_val = pd.DataFrame(scaler_X.transform(X_val_raw), columns=features)
y_train = pd.DataFrame(scaler_y.transform(y_train_raw), columns=target_col)
y_val = pd.DataFrame(scaler_y.transform(y_val_raw), columns=target_col)

###############################################################################
# === Model Definition (Random Forest) ===
rf_model = RandomForestRegressor(random_state=42)

# Parameter Grid
param_grid = {
    'n_estimators': [900, 1000, 1100],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# Grid Search
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='r2',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train.values.ravel())

###############################################################################
# === Evaluation ===
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)

r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)

print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"R²: {r2:.5f}")
print(f"MAE: {mae:.5f}")
print(f"RMSE: {rmse:.5f}")

###############################################################################
# === Residual Analysis ===
residuals = y_val.values.flatten() - y_pred.flatten()
res_mean = np.mean(residuals)
res_median = np.median(residuals)
hist_counts, bin_edges = np.histogram(residuals, bins=50)
mode_index = np.argmax(hist_counts)
res_mode = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2
lb_test = acorr_ljungbox(residuals, lags=[1], return_df=True)
prob_Q = lb_test["lb_pvalue"].iloc[0] if not lb_test.empty else np.nan

###############################################################################
# === Figures ===
figures = []

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(y_val.values[-720:], label='Real', color='gray')
ax1.plot(y_pred[-720:], label='Random Forest', color='lightgray', linestyle='--')
ax1.set_title(f'Random Forest Regression\nR²: {r2:.5f}')
ax1.legend(); ax1.grid(True)
fig1.tight_layout()
figures.append(fig1)

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.scatter(range(len(residuals)), residuals, alpha=0.5, s=10, color='gray')
ax2.axhline(0, color='black', linestyle='--')
ax2.set_title('Residual Errors'); ax2.grid(True)
fig2.tight_layout()
figures.append(fig2)

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.hist(residuals, bins=50, color='lightgray', edgecolor='gray')
ax3.axvline(res_mean, color='black', linestyle='--', label=f"Mean={res_mean:.5f}")
ax3.axvline(res_median, color='black', linestyle='-.', label=f"Median={res_median:.5f}")
ax3.axvline(res_mode, color='black', linestyle=':', label=f"Mode={res_mode:.5f}")
ax3.legend(); ax3.grid(True)
ax3.set_title('Residuals Distribution')
fig3.tight_layout()
figures.append(fig3)

# Feature importance
importance = pd.Series(best_model.feature_importances_, index=X_train.columns)
importance_sorted = importance.sort_values(ascending=True).tail(20)

fig4, ax4 = plt.subplots(figsize=(8, 5))
bars = ax4.barh(importance_sorted.index, importance_sorted.values, color='gray')
ax4.set_title('Top 20 Feature Importances (Random Forest)')
ax4.grid(True, axis='x', linestyle='--', alpha=0.7)
for bar in bars:
    width = bar.get_width()
    ax4.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
             f"{width:.3f}", va='center')
fig4.tight_layout()
figures.append(fig4)

plt.show()

###############################################################################
# === Save Report to PDF ===
report_file = f'randomforest_parameter_selection_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf'
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

params_info = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Model": "RandomForestRegressor",
    "Number of features used": len(features),
    "Train samples": len(y_train),
    "Validation samples": len(y_val),
}
for k, v in param_grid.items():
    params_info[f"Parameter Grid {k}"] = str(v)
for k, v in grid_search.best_params_.items():
    params_info[f"Best Parameter {k}"] = str(v)
params_info["R²"] = r2
params_info["MAE"] = mae
params_info["RMSE"] = rmse
params_info["Residuals Prob(Q)"] = prob_Q
params_info["Residuals skew"] = float(f"{pd.Series(residuals).skew():.3f}")
params_info["Residuals kurtosis"] = float(f"{pd.Series(residuals).kurtosis():.3f}")

save_report_to_pdf(
    filename=report_path,
    title=f"Random Forest Parameter Selection Report - {station_name}",
    params=params_info,
    features=features,
    errors=[(target_item, r2, mae, rmse)],
    figures=figures
)

print(f"\n✅ Report saved as {report_path}")
