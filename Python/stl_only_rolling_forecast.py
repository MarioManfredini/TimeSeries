# -*- coding: utf-8 -*-
"""
Created 2025/12/14
@author: Mario

STL-only Rolling Forecast (1-step ahead)
- Fixed training window (99%)
- Walk-forward forecast on validation set
- No residual modeling (baseline STL)
- Metrics: R², MAE, RMSE
- PDF report (same style as ARIMA reports)
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import STL
from scipy.stats import mode, skew, kurtosis

from utility import load_and_prepare_data, get_station_name
from report import save_report_to_pdf

warnings.filterwarnings("ignore")
plt.style.use("grayscale")

###############################################################################
# === Configuration ===
data_dir = Path("..") / "data" / "Ehime"
prefecture_code = "38"
station_code = "38206050"
station_name = get_station_name(data_dir, station_code)

target_item = "Ox(ppm)"
seasonal_period = 24

###############################################################################
def rolling_forecast_stl_only():
    """
    Perform STL-only rolling forecast (1-step ahead)
    and generate a PDF report.
    """

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    data, _ = load_and_prepare_data(
        data_dir, prefecture_code, station_code
    )

    series = data[target_item].dropna()
    series = series.asfreq("h")

    n_samples = len(series)
    train_size = int(0.99 * n_samples)
    val_series = series.iloc[train_size:]

    y_true = []
    y_pred = []

    # ------------------------------------------------------------------
    # 2. Rolling forecast loop
    # ------------------------------------------------------------------
    max_history = 24 * 30  # use last 30 days only (speed + stability)

    for i in range(len(val_series)):
        history = series.iloc[
            max(0, train_size + i - max_history) : train_size + i
        ]

        # --- STL decomposition ---
        stl = STL(
            history,
            period=seasonal_period,
            robust=True
        )
        stl_result = stl.fit()

        trend = stl_result.trend
        seasonal = stl_result.seasonal

        # --- STL-only reconstruction ---
        last_trend = trend.iloc[-1]
        next_seasonal = seasonal.iloc[-seasonal_period]

        forecast = last_trend + next_seasonal

        y_true.append(val_series.iloc[i])
        y_pred.append(forecast)

        print(f"- {i+1}/{len(val_series)}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ------------------------------------------------------------------
    # 3. Evaluation metrics
    # ------------------------------------------------------------------
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("--------------------------------------------------")
    print("Rolling Forecast: STL-only (1-step)")
    print("--------------------------------------------------")
    print(f"Station : {station_name}")
    print(f"Samples : {len(y_true)}")
    print("")
    print(f"R²   : {r2:.4f}")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print("--------------------------------------------------")

    # ------------------------------------------------------------------
    # 4. Residual analysis
    # ------------------------------------------------------------------
    residuals = y_true - y_pred

    res_mean = np.mean(residuals)
    res_median = np.median(residuals)
    res_mode = float(mode(residuals, nan_policy="omit").mode)

    skewness = skew(residuals)
    kurt = kurtosis(residuals, fisher=False)

    # ------------------------------------------------------------------
    # 5. Figures for report
    # ------------------------------------------------------------------
    figures = []

    # --- Forecast vs Observed ---
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(y_true[-720:], label="Observed", linewidth=0.8)
    ax1.plot(y_pred[-720:], label="STL-only forecast",
             linestyle="dashed", linewidth=0.8)
    ax1.set_title(f"STL-only Rolling Forecast (1-step)\nR²: {r2:.4f}")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel(target_item)
    ax1.legend()
    ax1.grid(True, linestyle="--", linewidth=0.3)
    fig1.tight_layout()
    figures.append(fig1)

    # --- Residuals scatter ---
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(range(len(residuals)), residuals,
                alpha=0.6, s=10)
    ax2.axhline(0, linestyle="--", linewidth=0.8)
    ax2.set_title("Residuals (Observed - Forecast)")
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Residual")
    ax2.grid(True, linestyle="--", linewidth=0.3)
    fig2.tight_layout()
    figures.append(fig2)

    # --- Residuals histogram ---
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.hist(residuals, bins=50, edgecolor="gray", alpha=0.75)
    ax3.axvline(res_mean, linestyle="--", linewidth=1,
                label=f"Mean = {res_mean:.5f}")
    ax3.axvline(res_median, linestyle="-.", linewidth=1,
                label=f"Median = {res_median:.5f}")
    ax3.axvline(res_mode, linestyle=":", linewidth=1,
                label=f"Mode = {res_mode:.5f}")
    ax3.set_title("Histogram of Residuals")
    ax3.set_xlabel("Residual value")
    ax3.set_ylabel("Frequency")
    ax3.legend()
    ax3.grid(True, linestyle="--", linewidth=0.3)
    fig3.tight_layout()
    figures.append(fig3)

    plt.show()

    # ------------------------------------------------------------------
    # 6. Parameters and errors summary
    # ------------------------------------------------------------------
    params_info = {
        "Prefecture code": prefecture_code,
        "Station code": station_code,
        "Station name": station_name,
        "Target item": target_item,
        "Model": "STL-only (rolling forecast)",
        "Seasonal period": seasonal_period,
        "Training proportion": "99%",
        "Training samples": train_size,
        "Validation samples": len(y_true),
        "Residuals mean": res_mean,
        "Residuals median": res_median,
        "Residuals mode": res_mode,
        "Residuals skewness": skewness,
        "Residuals kurtosis": kurt,
    }

    errors = [(target_item, r2, mae, rmse)]

    # ------------------------------------------------------------------
    # 7. Save report to PDF
    # ------------------------------------------------------------------
    report_file = (
        f"stl_only_rolling_forecast_{station_name}_"
        f"{prefecture_code}_{station_code}_{target_item}.pdf"
    )
    report_path = os.path.join("..", "reports", report_file)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    save_report_to_pdf(
        filename=report_path,
        title=f"STL-only Rolling Forecast Report - {station_name}",
        formula_image_path=None,
        params=params_info,
        features=[target_item],
        errors=errors,
        figures=figures
    )

    print(f"\n✅ Report saved as {report_path}")


###############################################################################
# === Run ===
###############################################################################
if __name__ == "__main__":
    rolling_forecast_stl_only()
