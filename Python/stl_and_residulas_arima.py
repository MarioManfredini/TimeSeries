# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 16:44:10 2025

@author: 471PC
"""

# --------------------------------------------------------------
# Rolling Forecast: STL + ARIMA (1-step ahead)
# --------------------------------------------------------------
# - Fixed training window (80%)
# - Walk-forward forecast on validation set
# - Metrics: R², MAE, RMSE
# --------------------------------------------------------------

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utility import load_and_prepare_data, get_station_name

plt.style.use("grayscale")

# --------------------------------------------------------------
# Parameters
# --------------------------------------------------------------
data_dir = Path("..") / "data" / "Ehime"
prefecture_code = "38"
station_code = "38206050"
station_name = get_station_name(data_dir, station_code)

target_item = "Ox(ppm)"
seasonal_period = 24

# ARIMA parameters for residuals
p, d, q = 8, 0, 1

# --------------------------------------------------------------
def rolling_forecast_stl_arima():
    try:
        # ------------------------------------------------------
        # 1. Load data
        # ------------------------------------------------------
        data, _ = load_and_prepare_data(
            data_dir, prefecture_code, station_code
        )

        series = data[target_item].dropna()
        series = series.asfreq("h")
        n_samples = len(series)

        # ------------------------------------------------------
        # 2. Train / Validation split
        # ------------------------------------------------------
        train_size = int(0.99 * n_samples)
        val_series = series.iloc[train_size:]

        y_true = []
        y_pred = []

        # ------------------------------------------------------
        # 3. Rolling forecast loop
        # ------------------------------------------------------
        for i in range(len(val_series)):
            # expanding window
            max_history = 24 * 30  # last days
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
            resid = stl_result.resid.dropna()

            # --- ARIMA on residuals ---
            model = ARIMA(resid, order=(p, d, q))
            model_fit = model.fit()

            resid_forecast = model_fit.forecast(steps=1).iloc[0]

            # --- reconstruct forecast ---
            last_trend = trend.iloc[-1]
            last_seasonal = seasonal.iloc[-seasonal_period]

            forecast = last_trend + last_seasonal + resid_forecast

            # store results
            y_true.append(val_series.iloc[i])
            y_pred.append(forecast)
            
            print(f"- {i}/{len(val_series)}")

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # ------------------------------------------------------
        # 4. Metrics
        # ------------------------------------------------------
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print("--------------------------------------------------")
        print("Rolling Forecast: STL + ARIMA (1-step)")
        print("--------------------------------------------------")
        print(f"Station : {station_name}")
        print(f"Samples : {len(y_true)}")
        print("")
        print(f"R²   : {r2:.4f}")
        print(f"MAE  : {mae:.6f}")
        print(f"RMSE : {rmse:.6f}")
        print("--------------------------------------------------")

        # ------------------------------------------------------
        # 5. Diagnostic plot
        # ------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(y_true, label="Observed", linewidth=0.8)
        ax.plot(y_pred, label="Forecast", linewidth=0.8)

        ax.set_title("STL + ARIMA Rolling Forecast (t+1)")
        ax.set_ylabel(target_item)
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.3)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error in rolling forecast: {e}")

# --------------------------------------------------------------
# Run
# --------------------------------------------------------------
if __name__ == "__main__":
    rolling_forecast_stl_arima()
