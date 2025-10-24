# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""
import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from utility import get_station_name, load_and_prepare_data
from report import save_report_to_pdf, plot_feature_summary_pages, plot_comparison_actual_predicted
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

start_time = time.time()

###############################################################################
# === Parameters ===
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'
forecast_horizon = 24  # n-step forecast

###############################################################################
def save_xgboost_formula_as_jpg(filename="formula_xgboost.jpg"):
    """
    Save XGBoost model formula and cost function as a JPEG image with explanation.
    """
    # Main prediction formula
    formula_prediction = (
        r"$\hat{y}_i = \sum_{m=1}^{M} \eta \, f_m(X_i)$"
    )

    # Cost (objective) function
    formula_cost = (
        r"$\mathcal{L} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \sum_{m=1}^{M} \Omega(f_m)$"
    )

    explanation_lines = [
        r"$\hat{y}_i$: predicted value for sample $i$",
        r"$M$: total number of trees (number of estimators, eg.:400)",
        r"$\eta$: learning rate (eg.:0.04), controlling the contribution of each tree",
        r"$f_m(x_i)$: output of the $m$-th regression tree with depth, eg.:\leq 6",
        r"$\mathcal{L}$: total loss function minimized during training",
        r"$\Omega(f_m)$: regularization term penalizing tree complexity (depth, leaf weights)",
        r"Objective = minimize the total cost: prediction error + regularization penalty.",
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis("off")

    # Draw formulas
    ax.text(0, 1, formula_prediction, fontsize=22, ha="left", va="top")
    ax.text(0, 0.78, formula_cost, fontsize=20, ha="left", va="top")

    # Explanation text
    y_start = 0.45
    line_spacing = 0.08
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * line_spacing, line,
                fontsize=12, ha="left", va="top")

    plt.tight_layout()

    # Save as temporary PNG
    temp_file = "_temp_formula_xgboost.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Convert to JPEG
    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved XGBoost formula image as {filename}")

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

# Create multi-output target
for i in range(forecast_horizon):
    df[f'{target_item}_t+{i+1:02}'] = df[target_item].shift(-i-1)

target_cols = [f'{target_item}_t+{i+1:02}' for i in range(forecast_horizon)]

data_model = df.dropna(subset=features + target_cols).copy()

###############################################################################
# === Normalization ===
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_array = scaler_X.fit_transform(data_model[features])
y_array = scaler_y.fit_transform(data_model[target_cols])

X = pd.DataFrame(X_array, columns=features)
y = pd.DataFrame(y_array, columns=target_cols)

###############################################################################
# === Train/Test Split ===
split_index = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

###############################################################################
# === Model Training ===
base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.045,
    max_depth=5,
    verbosity=0,
    random_state=42,
)

model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

###############################################################################
# === Forecast ===
X_test_df = pd.DataFrame(X_test, columns=features)
y_pred_scaled = model.predict(X_test_df)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

###############################################################################
# === Evaluation and Residuals ===
plt.figure(figsize=(12, 5))
t_index = data_model.index[split_index:split_index+len(y_pred)]

###############################################################################
# === Prints ===
print("Error per forecast step (t+01, t+02, ...):")
r2_list = []
mae_list = []
rmse_list = []
for i, col in enumerate(target_cols):
    r2_scores = r2_score(y_true[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    r2_list.append(r2_scores)
    mae_list.append(mae)
    rmse_list.append(rmse)
    print(f"{col}: R²: {r2_scores:.4f} MAE = {mae:.4f}, RMSE = {rmse:.4f}")

figures = []

#figs = plot_all_feature_importances(model.estimators_, target_cols, top_n=20, rows_per_page=6)
#figures.extend(figs)

#figs = plot_residuals_grid(y_true, y_pred, target_cols, targets_per_page=4)
#figures.extend(figs)

steps = [f't+{i:02}' for i in range(1, forecast_horizon + 1)]

figs = plot_comparison_actual_predicted(y_true, y_pred, target_cols, steps, rows_per_page=3)
figures.extend(figs)

figs = plot_feature_summary_pages(model, X_train.columns, target_cols, mae_list, rmse_list, r2_list, steps, top_n=15)
figures.extend(figs)

end_time = time.time()
elapsed_seconds = int(end_time - start_time)
elapsed_minutes = elapsed_seconds // 60
elapsed_seconds_remainder = elapsed_seconds % 60
elapsed_time_str = f"{elapsed_minutes} min {elapsed_seconds_remainder} sec"

# Save formula as image
formula_path = os.path.join("..", "reports", "formula_xgboost.jpg")
save_xgboost_formula_as_jpg(filename=formula_path)

explanation = (
    "The XGBoost model builds an ensemble of decision trees,\n"
    "where each tree tries to correct the residual errors made by the previous ones.\n"
    "At each iteration, the model adds a new function f(x) that minimizes\n"
    "a global objective function composed of two terms:\n"
    "1. The **prediction loss** (in this case, the squared error between actual and predicted values).\n"
    "2. A **regularization term** Ω(f) that penalizes complex trees to prevent overfitting.\n\n"
    "The squared error loss is used as the primary cost to minimize,\n"
    "while the regularization controls model complexity.\n"
    "The optimization proceeds additively, building the model tree by tree, guided by the learning rate."
)

model_params = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Number of data points in the train set": len(y_train),
    "Number of data points in the test set": len(y_test),
    "Forecast horizon (hours)": forecast_horizon,
    "Model": "XGBoost",
    "Objective": base_model.get_params()['objective'],
    "Booster": base_model.get_params().get('booster', 'gbtree'),
    "Number of estimators": base_model.get_params()['n_estimators'],
    "Learning rate": base_model.get_params()['learning_rate'],
    "Elapsed time": elapsed_time_str,
    "Number of features used": len(features),
}

errors = [
    (target_cols[i], r2_score(y_true[:, i], y_pred[:, i]), mae_list[i], rmse_list[i])
    for i in range(forecast_horizon)
]

# === Save PDF ===
report_file = f'forecast_xgboost_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf'
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_report_to_pdf(
    filename=report_path,
    title=f"XGBoost forecasting Report - {station_name}",
    formula_image_path=formula_path,
    comments=explanation,
    params=model_params,
    features=features,
    errors=errors,
    figures=figures,
)