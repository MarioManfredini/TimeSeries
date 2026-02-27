# -*- coding: utf-8 -*-
"""
Created 2026/02/23
NHITS 24h Rolling Forecast (Fixed Model, stride=1) + Historical Exogenous
@author: Mario
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from utility import load_and_prepare_data, get_station_name, add_solar_elevation
from report import (
    save_report_to_pdf,
    plot_comparison_actual_predicted,
    plot_error_summary_page
)

warnings.filterwarnings("ignore")
start_time = time.time()

###############################################################################
# === Parameters ===
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38205010'
station_name = get_station_name(data_dir, station_code)
station_latitude = 33.96278750333128
station_longitude = 133.27950978640163
timezone_offset = 9.0 # Japan

target_item = 'Ox(ppm)'
#hist_exog_features = ['NO(ppm)', 'NO2(ppm)', 'U', 'V', 'TEMP(℃)', 'HUM(％)']
hist_exog_features = ['solar_elevation']
futr_exog_features = hist_exog_features

forecast_horizon = 24
stride = 1

TRAIN_DAYS = 90
TEST_DAYS = 30

hours_per_day = 24
train_hours = TRAIN_DAYS * hours_per_day
test_hours = TEST_DAYS * hours_per_day

###############################################################################
def save_nhits_formula_as_jpg(filename="formula_nhits.jpg"):
    """
    Save NHITS model formula and loss function as a JPEG image with explanation.
    """

    formula_eq = (
        r"$\mathbf{x}_t \in \mathbb{R}^{L}$ (input window)" "\n"
        r"$\theta^{(b)} = f_{\mathrm{MLP}}^{(b)}(\mathbf{x}_t)$" "\n"
        r"$\hat{\mathbf{y}}^{(b)} = P^{(b)} \theta^{(b)}$" "\n"
        r"$\hat{\mathbf{y}} = \sum_{b=1}^{B} \hat{\mathbf{y}}^{(b)}$" "\n"
        r"$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$"
    )

    explanation_lines = [
        r"$\mathbf{x}_t$: input lookback window (L=720 hours)",
        r"$f_{\mathrm{MLP}}^{(b)}$: block-specific multi-layer perceptron",
        r"$\theta^{(b)}$: basis coefficients estimated by block $b$",
        r"$P^{(b)}$: hierarchical interpolation operator",
        r"$\hat{\mathbf{y}}$: 24-hour direct multi-step forecast",
        r"$\mathcal{L}$: Mean Absolute Error minimized during training",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    ax.text(0, 1, formula_eq,
            fontsize=17,
            ha="left",
            va="top",
            linespacing=1.6)

    y_start = 0.25
    line_spacing = 0.065

    for i, line in enumerate(explanation_lines):
        ax.text(
            0,
            y_start - i * line_spacing,
            line,
            fontsize=11,
            ha="left",
            va="top"
        )

    plt.tight_layout()

    temp_file = "_temp_formula_nhits.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    from PIL import Image
    import os

    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved NHITS formula image as {filename}")

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

df = df[[target_item]].dropna().copy()

df = df.reset_index()
df = df.rename(columns={"datetime": "ds", target_item: "y"})
df["unique_id"] = "series_1"

df = add_solar_elevation(
    df,
    datetime_col="ds",
    latitude=station_latitude,
    longitude=station_longitude,
    timezone_offset=timezone_offset,
    normalize=True,
    clip_night=True
)

df = df[["unique_id", "ds", "y"] + hist_exog_features]
df = df.sort_values("ds")

###############################################################################
# === Train/Test Split ===
df_train = df.iloc[-(train_hours + test_hours):-test_hours].copy()
df_test = df.iloc[-test_hours:].copy()

###############################################################################
# === Model Definition ===
model = NHITS(
    h=forecast_horizon,
    input_size=720,
    max_steps=1500,
    learning_rate=1e-3,
    n_blocks=[1,1,1],
    mlp_units=[[128,128],[128,128],[128,128]],
    stack_types=["identity","identity","identity"],
    hist_exog_list=hist_exog_features,
    futr_exog_list=futr_exog_features
)

nf = NeuralForecast(
    models=[model],
    freq='H'
)

###############################################################################
# === Fit Once ===
nf.fit(df_train)

###############################################################################
# === Rolling Evaluation ===
y_preds = []
y_trues = []

n_test = len(df_test)

for start in range(0, n_test - forecast_horizon + 1, stride):

    df_context = pd.concat([
        df_train,
        df_test.iloc[:start]
    ])

    # ---- 1. last timestamp ----
    last_ds = df_context["ds"].max()

    # ---- 2. create future timestamps ----
    future_dates = pd.date_range(
        start=last_ds + pd.Timedelta(hours=1),
        periods=forecast_horizon,
        freq="H"
    )

    # ---- 3. build futr_df ----
    futr_df = pd.DataFrame({
        "unique_id": "series_1",
        "ds": future_dates
    })

    # ---- 4. compute future solar elevation ----
    futr_df = add_solar_elevation(
        futr_df,
        datetime_col="ds",
        latitude=station_latitude,
        longitude=station_longitude,
        timezone_offset=timezone_offset,
        normalize=True,
        clip_night=True
    )

    # Keep required columns
    futr_df = futr_df[["unique_id", "ds", "solar_elevation"]]

    # ---- 5. forecast ----
    forecast_df = nf.predict(df_context, futr_df=futr_df)

    forecast_values = forecast_df["NHITS"].values
    true_values = df_test.iloc[start:start+forecast_horizon]["y"].values

    y_preds.append(forecast_values)
    y_trues.append(true_values)

    print(f"- Rolling window {(start//stride)+1}")

y_preds = np.array(y_preds)
y_trues = np.array(y_trues)

###############################################################################
# === Evaluation per Horizon ===
r2_list, mae_list, rmse_list = [], [], []

for h in range(forecast_horizon):
    y_true_h = y_trues[:, h]
    y_pred_h = y_preds[:, h]

    r2_list.append(r2_score(y_true_h, y_pred_h))
    mae_list.append(mean_absolute_error(y_true_h, y_pred_h))
    rmse_list.append(np.sqrt(mean_squared_error(y_true_h, y_pred_h)))

    print(
        f"t+{h+1:02d} "
        f"R²={r2_list[-1]:.4f} "
        f"MAE={mae_list[-1]:.4f} "
        f"RMSE={rmse_list[-1]:.4f}"
    )

###############################################################################
# === Plots ===
figures = []
steps = [f"t+{i:02d}" for i in range(1, forecast_horizon + 1)]

figs = plot_comparison_actual_predicted(
    y_trues,
    y_preds,
    target_cols=steps,
    steps=steps,
    rows_per_page=3
)
figures.extend(figs)

figs = plot_error_summary_page(mae_list, rmse_list, r2_list, steps)
figures.append(figs)

###############################################################################
# === Report ===
elapsed = int(time.time() - start_time)

formula_path = os.path.join("..", "reports", "formula_nhits.jpg")
save_nhits_formula_as_jpg(filename=formula_path)

explanation = (
    "The N-HiTS model is configured using historical exogenous variables "
    "(NO, NO2 concentrations and wind components U, V).\n\n"
    "These variables are provided only from past observations and allow the "
    "model to learn dynamic interactions between atmospheric chemistry and "
    "meteorological transport mechanisms.\n\n"
    "No manual lag engineering is applied, as the model directly processes "
    "a 720-hour historical window."
)

params = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Target item": target_item,
    "Model": "NHITS (Fixed, Rolling stride=1)",
    "Forecast horizon": forecast_horizon,
    "Input size": 720,
    "Historical exogenous": ", ".join(hist_exog_features),
    "Future exogenous": ", ".join(futr_exog_features),
    "Training samples": len(df_train),
    "Test samples": len(df_test),
    "Elapsed time": f"{elapsed // 60} min {elapsed % 60} sec",
}

errors = [
    (steps[i], r2_list[i], mae_list[i], rmse_list[i])
    for i in range(forecast_horizon)
]

report_path = os.path.join(
    "..", "reports",
    f"nhits_24h_hist_futr_exog_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf"
)

save_report_to_pdf(
    filename=report_path,
    title=f"NHITS 24-hour Forecast Report - {station_name}",
    formula_image_path=formula_path,
    comments=explanation,
    params=params,
    features=hist_exog_features,
    errors=errors,
    figures=figures
)

print(f"\n✅ NHITS 24h forecast report saved: {report_path}")