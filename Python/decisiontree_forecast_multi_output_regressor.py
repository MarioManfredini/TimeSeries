# -*- coding: utf-8 -*-
"""
Created 2025/11/09

@author: Mario
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utility import get_station_name, load_and_prepare_data
from report import save_report_to_pdf, plot_feature_summary_pages, plot_comparison_actual_predicted

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
def save_decision_tree_formula_as_jpg(filename="formula_decision_tree.jpg"):
    """
    Save Decision Tree model formula and cost function as a JPEG image with explanation.
    """
    # Prediction and cost function
    formula_prediction = (
        r"$\hat{y}(x) = \frac{1}{N_j} \sum_{x_i \in R_j} y_i \quad \text{for } x \in R_j$"
    )

    formula_cost = (
        r"$\mathcal{L} = \sum_{j=1}^{J} \sum_{x_i \in R_j} (y_i - \hat{y}_j)^2$"
    )

    explanation_lines = [
        r"Each node partitions the input space into regions $R_j$.",
        r"In each region, prediction $\hat{y}_j$ is the average of training targets.",
        r"The tree is built by recursively minimizing the total squared error $\mathcal{L}$.",
        r"Splits are chosen to maximize the reduction in variance of target values.",
        r"The depth and leaf size control overfitting vs generalization.",
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis("off")

    ax.text(0, 1, formula_prediction, fontsize=22, ha="left", va="top")
    ax.text(0, 0.78, formula_cost, fontsize=20, ha="left", va="top")

    y_start = 0.45
    line_spacing = 0.08
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * line_spacing, line, fontsize=12, ha="left", va="top")

    plt.tight_layout()

    temp_file = "_temp_formula_decision_tree.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved Decision Tree formula image as {filename}")

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Feature Engineering ===
features = ['NO(ppm)', 'NO2(ppm)', 'U', 'V']
lagged_items = [target_item, 'NO(ppm)', 'NO2(ppm)', 'U', 'V']

# Lag features
for item in lagged_items:
    for l in range(1, 24):
        df[f'{item}_lag{l}'] = df[item].shift(l)
        features.append(f'{item}_lag{l}')

# Rolling mean/std
for item in lagged_items:
    df[f'{item}_roll_mean_3'] = df[item].shift(1).rolling(3).mean()
    df[f'{item}_roll_std_6'] = df[item].shift(1).rolling(6).std()
    features += [f'{item}_roll_mean_3', f'{item}_roll_std_6']

# Differences
df[f'{target_item}_diff_1'] = df[target_item].shift(1).diff(1)
df[f'{target_item}_diff_2'] = df[target_item].shift(1).diff(2)
df[f'{target_item}_diff_3'] = df[target_item].shift(1).diff(3)
features += [f'{target_item}_diff_1', f'{target_item}_diff_2', f'{target_item}_diff_3']

for item in ['NO(ppm)', 'NO2(ppm)', 'U', 'V']:
    df[f'{item}_diff_3'] = df[item].shift(1).diff(3)
    features.append(f'{item}_diff_3')

# Time features
df["hour_sin"] = np.sin(2 * np.pi * df["時"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["時"] / 24)
df["dayofweek"] = df.index.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
features += ['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend']

# Multi-output target (t+1, ..., t+24)
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
base_model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    splitter="best",
    random_state=42
)

model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

###############################################################################
# === Forecast ===
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

###############################################################################
# === Evaluation ===
print("Error per forecast step (t+01, t+02, ...):")
r2_list, mae_list, rmse_list = [], [], []

for i, col in enumerate(target_cols):
    r2s = r2_score(y_true[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    r2_list.append(r2s)
    mae_list.append(mae)
    rmse_list.append(rmse)
    print(f"{col}: R²={r2s:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

###############################################################################
# === Plots for Report ===
figures = []
steps = [f't+{i:02}' for i in range(1, forecast_horizon + 1)]

figs = plot_comparison_actual_predicted(y_true, y_pred, target_cols, steps, rows_per_page=3)
figures.extend(figs)

figs = plot_feature_summary_pages(model, X_train.columns, target_cols, mae_list, rmse_list, r2_list, steps, top_n=15)
figures.extend(figs)

###############################################################################
# === Save formula image ===
formula_path = os.path.join("..", "reports", "formula_decision_tree.jpg")
save_decision_tree_formula_as_jpg(filename=formula_path)

###############################################################################
# === Report info ===
end_time = time.time()
elapsed_seconds = int(end_time - start_time)
elapsed_time_str = f"{elapsed_seconds // 60} min {elapsed_seconds % 60} sec"

explanation = (
    "A Decision Tree splits the input space into regions (R₁, R₂, …, Rj) by recursively partitioning the data\n"
    "along the features that most reduce the variance of the target variable.\n"
    "For each region Rj, the model predicts the average of training samples within that region.\n"
    "The cost function minimized at each split is the total squared error L.\n"
    "Tree depth and leaf size control model complexity, balancing bias and variance."
)

model_params = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Forecast horizon (hours)": forecast_horizon,
    "Number of training samples": len(y_train),
    "Number of test samples": len(y_test),
    "Number of features used": len(features),
    "Model": "DecisionTreeRegressor",
    "Max depth": base_model.get_params()['max_depth'],
    "Min samples split": base_model.get_params()['min_samples_split'],
    "Min samples leaf": base_model.get_params()['min_samples_leaf'],
    "Elapsed time": elapsed_time_str,
}

errors = [
    (target_cols[i], r2_list[i], mae_list[i], rmse_list[i])
    for i in range(forecast_horizon)
]

###############################################################################
# === Save Report ===
report_file = f'decisiontree_forecast_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf'
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_report_to_pdf(
    filename=report_path,
    title=f"Decision Tree Forecasting Report - {station_name}",
    formula_image_path=formula_path,
    comments=explanation,
    params=model_params,
    features=features,
    errors=errors,
    figures=figures,
)

print(f"\n✅ Report saved as {report_path}")
