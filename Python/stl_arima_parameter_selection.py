# -*- coding: utf-8 -*-
"""
Created 2025/12/14
@author: Mario
Rolling Forecast STL + ARIMA with Grid Search and PDF Report
"""

# --------------------------------------------------------------
# Imports
# --------------------------------------------------------------
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import skew, kurtosis, mode

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

from utility import load_and_prepare_data, get_station_name
from report import save_report_to_pdf

warnings.filterwarnings("ignore")
plt.style.use("grayscale")

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
data_dir = Path("..") / "data" / "Ehime"
prefecture_code = "38"
station_code = "38206050"
station_name = get_station_name(data_dir, station_code)

target_item = "Ox(ppm)"
seasonal_period = 24

train_ratio = 0.99
max_history = 24 * 30   # last 30 days used for STL

# ARIMA grid (residuals)
p_values = [2, 4, 6, 8]
d_values = [0]
q_values = [0, 1, 2]

# --------------------------------------------------------------
def save_stl_arima_formula_as_jpg(filename="formula_stl_arima.jpg"):
    """
    Save STL + ARIMA model formula and explanation as JPEG.
    """

    formula = (
        r"$y_t = T_t + S_t + R_t$" "\n"
        r"$R_t \sim \mathrm{ARIMA}(p,d,q)$"
    )

    explanation = [
        r"$y_t$: observed time series",
        r"$T_t$: trend component (STL)",
        r"$S_t$: seasonal component (STL)",
        r"$R_t$: residual component modeled by ARIMA",
        r"Final forecast = trend + seasonal + ARIMA(residual)",
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    ax.text(0, 1, formula, fontsize=22, ha="left", va="top")

    y_start = 0.55
    for i, line in enumerate(explanation):
        ax.text(0, y_start - i * 0.08, line, fontsize=11, ha="left", va="top")

    plt.tight_layout()

    tmp = "_tmp_formula.png"
    fig.savefig(tmp, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    img = Image.open(tmp).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(tmp)

# --------------------------------------------------------------
def rolling_forecast_stl_arima(series, order):
    """
    Rolling 1-step forecast using STL + ARIMA(residuals)
    """

    n_samples = len(series)
    train_size = int(train_ratio * n_samples)
    val_series = series.iloc[train_size:]

    y_true, y_pred = [], []

    for i in range(len(val_series)):
        history = series.iloc[
            max(0, train_size + i - max_history): train_size + i
        ]

        stl = STL(history, period=seasonal_period, robust=True)
        res = stl.fit()

        resid = res.resid.dropna()

        model = ARIMA(resid, order=order)
        model_fit = model.fit()

        resid_fc = model_fit.forecast(steps=1).iloc[0]

        trend_last = res.trend.iloc[-1]
        seasonal_last = res.seasonal.iloc[-seasonal_period]

        forecast = trend_last + seasonal_last + resid_fc

        y_true.append(val_series.iloc[i])
        y_pred.append(forecast)

    return np.array(y_true), np.array(y_pred)

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)
series = df[target_item].dropna()
series = series.asfreq("h")

# --------------------------------------------------------------
# Grid Search
# --------------------------------------------------------------
best_r2 = -np.inf
best_order = None
grid_results = []

print("Starting STL + ARIMA grid search...")

for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                y_true, y_pred = rolling_forecast_stl_arima(
                    series, (p, d, q)
                )
                r2 = r2_score(y_true, y_pred)
                grid_results.append(((p, d, q), r2))

                print(f"Tested ARIMA{(p,d,q)} -> R² = {r2:.4f}")

                if r2 > best_r2:
                    best_r2 = r2
                    best_order = (p, d, q)

            except Exception:
                continue

print("\nBest STL + ARIMA order found:", best_order)

# --------------------------------------------------------------
# Final evaluation with best parameters
# --------------------------------------------------------------
y_true, y_pred = rolling_forecast_stl_arima(series, best_order)

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

residuals = y_true - y_pred

# Statistics
res_mean = np.mean(residuals)
res_median = np.median(residuals)
res_mode = float(mode(residuals, nan_policy="omit").mode)
skewness = skew(residuals)
kurt = kurtosis(residuals, fisher=False)

lb_test = acorr_ljungbox(residuals, lags=[1], return_df=True)
prob_Q = lb_test["lb_pvalue"].iloc[0]

# --------------------------------------------------------------
# Figures for report
# --------------------------------------------------------------
figures = []

# Forecast vs observed
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(y_true[-720:], label="Observed", color="gray")
ax1.plot(y_pred[-720:], label="STL + ARIMA forecast", linestyle="dashed")
ax1.set_title(f"STL + ARIMA{best_order}\nR² = {r2:.4f}")
ax1.set_ylabel(target_item)
ax1.legend()
ax1.grid(True)
fig1.tight_layout()
figures.append(fig1)

# Residuals
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.scatter(range(len(residuals)), residuals, s=10, alpha=0.6)
ax2.axhline(0, linestyle="--")
ax2.set_title("Residuals")
ax2.set_xlabel("Samples")
ax2.set_ylabel("Residual")
ax2.grid(True)
fig2.tight_layout()
figures.append(fig2)

# Histogram
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.hist(residuals, bins=50, edgecolor="gray", alpha=0.75)
ax3.axvline(res_mean, linestyle="--", label=f"Mean = {res_mean:.5f}")
ax3.axvline(res_median, linestyle="-.", label=f"Median = {res_median:.5f}")
ax3.axvline(res_mode, linestyle=":", label=f"Mode = {res_mode:.5f}")
ax3.set_title("Residuals Distribution")
ax3.legend()
ax3.grid(True)
fig3.tight_layout()
figures.append(fig3)

plt.show()

# --------------------------------------------------------------
# Save formula
# --------------------------------------------------------------
formula_path = os.path.join("..", "reports", "formula_stl_arima.jpg")
save_stl_arima_formula_as_jpg(formula_path)

# --------------------------------------------------------------
# Report parameters
# --------------------------------------------------------------
params_info = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Model": "STL + ARIMA",
    "Seasonal period": seasonal_period,
    "Training ratio": train_ratio,
    "Max STL history": max_history,
    "Best ARIMA order": best_order,
    "Residuals Ljung-Box Prob(Q)": prob_Q,
    "Residuals skewness": skewness,
    "Residuals kurtosis": kurt,
}

for key, values in {
    "p": p_values, "d": d_values, "q": q_values
}.items():
    params_info[f"Grid {key}"] = str(values)

errors = [(target_item, r2, mae, rmse)]

# --------------------------------------------------------------
# Save PDF report
# --------------------------------------------------------------
report_file = (
    f"stl_arima_parameter_selection_"
    f"{station_name}_{prefecture_code}_{station_code}.pdf"
)
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_report_to_pdf(
    filename=report_path,
    title=f"STL + ARIMA parameter selection Report - {station_name}",
    formula_image_path=formula_path,
    params=params_info,
    features=[target_item],
    errors=errors,
    figures=figures,
)

print(f"\n✅ Report saved as {report_path}")
