# -*- coding: utf-8 -*-
"""
Created 2026/02/21
@author: Mario
N-HiTS model testing module
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import skew, kurtosis, mode

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from utility import load_and_prepare_data, get_station_name
from report import save_report_to_pdf

warnings.filterwarnings("ignore")

###############################################################################
# === Configuration ===
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'

H = 24  # forecast horizon

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)
df = df[[target_item]].dropna()

df = df.reset_index()
df = df.rename(columns={"datetime": "ds", target_item: "y"})
df["unique_id"] = "series_1"

###############################################################################
# === Train/Test Split ===
TRAIN_DAYS = 365
TEST_DAYS = 30

hours_per_day = 24
train_hours = TRAIN_DAYS * hours_per_day
test_hours = TEST_DAYS * hours_per_day

df_train = df.iloc[-(train_hours + test_hours):-test_hours]
df_test  = df.iloc[-test_hours:]

###############################################################################
# === Define N-HiTS model ===
model = NHITS(
    h=H,
    input_size=720, # 1 month lookback
    max_steps=1500,
    learning_rate=1e-3,
    n_blocks=[1,1,1],
    mlp_units=[[128,128],[128,128],[128,128]],
    stack_types=["identity","identity","identity"]
)

nf = NeuralForecast(models=[model], freq='H')

###############################################################################
# === Fit Model ===
nf.fit(df_train)

###############################################################################
# === Forecast rolling on test set ===
forecasts = nf.predict()

# Align prediction to test horizon
y_pred = forecasts["NHITS"].values[:len(df_test)]
y_test = df_test["y"].values[:len(y_pred)]

###############################################################################
# === Evaluation metrics ===
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R²: {r2:.5f}")
print(f"MAE: {mae:.5f}")
print(f"RMSE: {rmse:.5f}")

residuals = y_test - y_pred

res_mean = np.mean(residuals)
res_median = np.median(residuals)
res_mode = float(mode(residuals).mode)

skewness = skew(residuals)
kurt = kurtosis(residuals, fisher=False)

###############################################################################
# === Figures ===
figures = []

# Predicted vs Real
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(y_test[-720:], label='Real values', color='gray')
ax1.plot(y_pred[-720:], label='N-HiTS forecast', linestyle='dashed')
ax1.set_title(f'N-HiTS (H=24)\nR²: {r2:.5f}')
ax1.set_xlabel('Samples')
ax1.set_ylabel(target_item)
ax1.legend()
ax1.grid(True)
fig1.tight_layout()
figures.append(fig1)

# Residuals
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.scatter(range(len(residuals)), residuals, alpha=0.6, s=10)
ax2.axhline(0, linestyle='--')
ax2.set_title('Residuals')
ax2.set_xlabel('Samples')
ax2.set_ylabel('Residual (Real - Predicted)')
ax2.grid(True)
fig2.tight_layout()
figures.append(fig2)

# Histogram residuals
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.hist(residuals, bins=50, alpha=0.75)
ax3.axvline(res_mean, linestyle='--', linewidth=1, label=f'Mean = {res_mean:.5f}')
ax3.axvline(res_median, linestyle='-.', linewidth=1, label=f'Median = {res_median:.5f}')
ax3.axvline(res_mode, linestyle=':', linewidth=1, label=f'Mode = {res_mode:.5f}')
ax3.set_title('Histogram of Residuals')
ax3.set_xlabel('Residual value')
ax3.set_ylabel('Frequency')
ax3.legend()
fig3.tight_layout()
figures.append(fig3)

plt.show()

###############################################################################
# === Parameters and Errors Summary ===
params_info = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Number of training samples": len(df_train),
    "Number of testing samples": len(df_test),
    "Model": "N-HiTS",
    "Horizon (H)": H,
    "Input size": 168,
}

params_info["Predictions mean"] = np.mean(y_pred)
params_info["Predictions std"] = np.std(y_pred)
params_info["Real mean"] = np.mean(y_test)
params_info["Real std"] = np.std(y_test)

params_info["Residuals skew"] = skewness
params_info["Residuals kurtosis"] = kurt

errors = [(target_item, r2, mae, rmse)]

###############################################################################
# === Save Report ===
report_file = f'nhits_report_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf'
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_report_to_pdf(
    filename=report_path,
    title=f"N-HiTS Report - {station_name}",
    formula_image_path=None,
    params=params_info,
    features=[target_item],
    errors=errors,
    figures=figures
)

print(f"\n✅ Report saved as {report_path}")
