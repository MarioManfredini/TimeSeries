# -*- coding: utf-8 -*-
"""
Created 2025/10/19
@author: Mario
ARIMA model testing module
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import mode, skew, kurtosis
from PIL import Image

from utility import load_and_prepare_data, get_station_name, sanitize_filename_component
from report import save_report_to_pdf

warnings.filterwarnings("ignore")

###############################################################################
# === Configuration ===
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'

###############################################################################
def save_arima_formula_as_jpg(filename="formula_arima.jpg"):
    """
    Save ARIMA model formula and explanation as a JPEG image.
    """

    # --- Main ARIMA model formula ---
    formula_arima = (
        r"$\text{ARIMA}(p,d,q):$"
        "\n"
        r"$(1 - \sum_{i=1}^{p} \phi_i L^i)(1 - L)^d y_t = c + (1 + \sum_{i=1}^{q} \theta_i L^i)\varepsilon_t$"
    )

    # --- Forecast formula ---
    formula_forecast = (
        r"$\hat{y}_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j}$"
    )

    # --- Explanation text ---
    explanation_lines = [
        r"$y_t$: observed value at time $t$",
        r"$\hat{y}_t$: predicted (fitted) value at time $t$",
        r"$\phi_i$: autoregressive (AR) coefficients, capturing dependence on past values",
        r"$\theta_j$: moving average (MA) coefficients, capturing dependence on past errors",
        r"$d$: differencing order for stationarity (number of times data are differenced)",
        r"$L$: lag operator ($L y_t = y_{t-1}$)",
        r"$\varepsilon_t$: white noise (random shock) at time $t$",
        r"$c$: constant or drift term",
        r"Model objective = minimize residual variance $\sigma^2_{\varepsilon}$ to fit observed series.",
    ]

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    # --- Draw main formulas ---
    ax.text(0, 1, formula_arima, fontsize=20, ha="left", va="top")
    ax.text(0, 0.72, formula_forecast, fontsize=18, ha="left", va="top")

    # --- Explanations ---
    y_start = 0.48
    line_spacing = 0.07
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * line_spacing, line, fontsize=11, ha="left", va="top")

    plt.tight_layout()

    # --- Save as temporary PNG ---
    temp_file = "_temp_formula_arima.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # --- Convert to JPEG ---
    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved ARIMA formula image as {filename}")

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

# Keep only target column and drop NaN
df = df[[target_item]].dropna()

###############################################################################
# === Train/Test Split ===
# Train/test split
TRAIN_DAYS = 365
TEST_DAYS = 30

hours_per_day = 24
train_hours = TRAIN_DAYS * hours_per_day
test_hours = TEST_DAYS * hours_per_day

y_train = df.iloc[-(train_hours + test_hours):-test_hours]
y_test  = df.iloc[-test_hours:]

###############################################################################
# === Grid Search for ARIMA (p, d, q) ===
p_values = [0, 1, 2]
d_values = [0, 1, 2]
q_values = [1, 2, 3]

param_grid = {"p": p_values, "d": d_values, "q": q_values}

best_score = -np.inf
best_order = None
results = []

# Manual grid search over p, d, q
for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                model = ARIMA(y_train, order=(p, d, q))
                model_fit = model.fit()
                y_pred = model_fit.forecast(steps=len(y_test))
                r2 = r2_score(y_test, y_pred)
                results.append(((p, d, q), r2))
                print(f"Parameters: (p, d, q)=({p},{d},{q}) - r2={r2}")
                if r2 > best_score:
                    best_score = r2
                    best_order = (p, d, q)
            except Exception:
                continue

print("Best ARIMA parameters found:", best_order)

###############################################################################
# === Fit best ARIMA model ===
best_model = ARIMA(y_train, order=best_order)
best_model_fit = best_model.fit()
y_pred = best_model_fit.forecast(steps=len(y_test))

###############################################################################
# === Evaluation metrics ===
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R²: {r2:.5f}")
print(f"MAE: {mae:.5f}")
print(f"RMSE: {rmse:.5f}")

print("NaN in y_test:", y_test.isna().sum().values)
print("NaN in y_pred:", y_pred.isna().sum())
print("y_test shape:", y_test.shape)
print("y_pred shape:", y_pred.shape)
print("Index equal:", y_test.index.equals(y_pred.index))

y_test_np = y_test.squeeze().values
y_pred_np = y_pred.values

mask = np.isfinite(y_test_np) & np.isfinite(y_pred_np)
residuals = y_test_np[mask] - y_pred_np[mask]

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

# Predicted vs Real (last 720 samples)
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(y_test.values[-720:], label='Real values', color='gray')
ax1.plot(y_pred.values[-720:], label='ARIMA forecast', color='lightgray', linestyle='dashed')
ax1.set_title(f'ARIMA{best_order}\nR²: {r2:.5f}')
ax1.set_xlabel('Samples')
ax1.set_ylabel(target_item)
ax1.legend()
ax1.grid(True)
fig1.tight_layout()
figures.append(fig1)

# Residuals plot
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.scatter(range(len(residuals)), residuals, color='gray', alpha=0.6, s=10)
ax2.axhline(0, color='gray', linestyle='--')
ax2.set_title('Residuals')
ax2.set_xlabel('Samples')
ax2.set_ylabel('Residual (Real - Predicted)')
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

plt.show()

###############################################################################
# Save formula as image
formula_path = os.path.join("..", "reports", "formula_arima.jpg")
save_arima_formula_as_jpg(filename=formula_path)

###############################################################################
# === Parameters and Errors Summary ===
params_info = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Number of training samples": len(y_train),
    "Number of testing samples": len(y_test),
    "Model": "ARIMA",
    "ARIMA order": best_order,
}

# Tested parameter grid
for key, values in param_grid.items():
    params_info[f"Parameter Grid (tested) {key}"] = str(values)

# Best parameters
params_info["Best Parameters (found)"] = f"p={best_order[0]}, d={best_order[1]}, q={best_order[2]}"

params_info["Predictions mean"] = np.mean(y_pred)
params_info["Predictions std"] = np.std(y_pred)
params_info["Real mean"] = np.mean(y_test.values)
params_info["Real std"] = np.std(y_test.values)

params_info["Ljung-Box residuals autocorrelation, Prob(Q)"] = prob_Q

params_info["Residuals skew"] = skewness
params_info["Residuals kurtosis"] = kurt

errors = [(target_item, r2, mae, rmse)]

###############################################################################
# === Save Report to PDF ===
safe_target_item = sanitize_filename_component(target_item)
report_file = f'arima_parameter_selection_{station_name}_{prefecture_code}_{station_code}_{safe_target_item}.pdf'
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_report_to_pdf(
    filename=report_path,
    title=f"ARIMA parameter selection Report - {station_name}",
    formula_image_path=formula_path,
    params=params_info,
    features=[target_item],
    errors=errors,
    figures=figures
)

print(f"\n✅ Report saved as {report_path}")
