# -*- coding: utf-8 -*-
"""
Created 2025/10/19
@author: Mario
SARIMA model testing module
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import mode, skew, kurtosis
from PIL import Image

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

###############################################################################
def save_sarima_formula_as_jpg(filename="formula_sarima.jpg"):
    """
    Save SARIMA model formula and explanation as a JPEG image.
    """
    formula_sarima = (
        r"$\text{SARIMA}(p,d,q)(P,D,Q)_s:$"
        "\n"
        r"$(1 - \sum_{i=1}^{p}\phi_i L^i)"
        r"(1 - \sum_{I=1}^{P}\Phi_I L^{sI})"
        r"(1 - L)^d(1 - L^s)^D y_t =$"
        "\n"
        r"$c + (1 + \sum_{i=1}^{q}\theta_i L^i)"
        r"(1 + \sum_{I=1}^{Q}\Theta_I L^{sI})\varepsilon_t$"
    )

    formula_forecast = (
        r"$\hat{y}_t = c + \sum_{i=1}^{p}\phi_i y_{t-i} + "
        r"\sum_{I=1}^{P}\Phi_I y_{t-sI} + "
        r"\sum_{j=1}^{q}\theta_j \varepsilon_{t-j} + "
        r"\sum_{J=1}^{Q}\Theta_J \varepsilon_{t-sJ}$"
    )

    explanation_lines = [
        r"$y_t$: observed value at time $t$",
        r"$\hat{y}_t$: predicted value at time $t$",
        r"$\phi_i, \Phi_I$: non-seasonal and seasonal AR coefficients",
        r"$\theta_j, \Theta_J$: non-seasonal and seasonal MA coefficients",
        r"$d, D$: differencing orders (non-seasonal and seasonal)",
        r"$s$: seasonal period (here 24 for hourly data)",
        r"$L$: lag operator ($L y_t = y_{t-1}$)",
        r"$\varepsilon_t$: white noise (random error term)",
        r"Objective = minimize residual variance $\sigma^2_{\varepsilon}$ to fit observed series.",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.text(0, 1, formula_sarima, fontsize=14, ha="left", va="top")
    ax.text(0, 0.6, formula_forecast, fontsize=14, ha="left", va="top")

    y_start = 0.4
    line_spacing = 0.07
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * line_spacing, line, fontsize=9, ha="left", va="top")

    plt.tight_layout()
    temp_file = "_temp_formula_sarima.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)
    print(f"✅ Saved SARIMA formula image as {filename}")

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)
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
# === Non-seasonal grid ===
p_values = [0, 1, 2]
d_values = [0]
q_values = [0, 1, 2]

param_grid = {"p": p_values, "d": d_values, "q": q_values}

# === Fixed seasonal parameters ===
P, D, Q, s = 1, 1, 1, 24

best_score = -np.inf
best_order = None
results = []

# --- Grid search over non-seasonal (p,d,q) ---
for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                model = SARIMAX(
                    y_train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_fit = model.fit(disp=False)
                y_pred = model_fit.forecast(steps=len(y_test))
                r2 = r2_score(y_test, y_pred)
                results.append(((p, d, q), r2))
                print(f"Non-seasonal parameters: (p, d, q)=({p},{d},{q}) - r2={r2}")
                if r2 > best_score:
                    best_score = r2
                    best_order = (p, d, q)
            except Exception:
                continue

print("Best SARIMA non-seasonal parameters found:", best_order)
print(f"Seasonal part fixed at: (P,D,Q,s)=({P},{D},{Q},{s})")

###############################################################################
# === Fit best SARIMA model ===
best_model = SARIMAX(
    y_train,
    order=best_order,
    seasonal_order=(P, D, Q, s),
    enforce_stationarity=False,
    enforce_invertibility=False
)
best_model_fit = best_model.fit(disp=False)
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

# Predicted vs Real
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(y_test.values[-720:], label='Real values', color='gray')
ax1.plot(y_pred.values[-720:], label='SARIMA forecast', color='lightgray', linestyle='dashed')
ax1.set_title(f'SARIMA{best_order}x(P=1,D=1,Q=1,s=24)\nR²: {r2:.5f}')
ax1.set_xlabel('Samples')
ax1.set_ylabel(target_item)
ax1.legend()
ax1.grid(True)
fig1.tight_layout()
figures.append(fig1)

# Residuals
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
formula_path = os.path.join("..", "reports", "formula_sarima.jpg")
save_sarima_formula_as_jpg(filename=formula_path)

###############################################################################
# === Parameters and Errors Summary ===
params_info = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Number of training samples": len(y_train),
    "Number of testing samples": len(y_test),
    "Model": "SARIMA",
    "SARIMA order": best_order,
    "Seasonal order": "(P=1,D=1,Q=1,s=24)",
}

# Tested parameter grid
for key, values in param_grid.items():
    params_info[f"Parameter Grid (tested) {key}"] = str(values)

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
report_file = f'sarima_parameter_selection_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf'
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_report_to_pdf(
    filename=report_path,
    title=f"SARIMA parameter selection Report - {station_name}",
    formula_image_path=formula_path,
    params=params_info,
    features=[target_item],
    errors=errors,
    figures=figures
)

print(f"\n✅ Report saved as {report_path}")
