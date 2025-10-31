# -*- coding: utf-8 -*-
"""
Created 2025/10/26
@author: Mario
SARIMAX model testing module (uses exogenous features)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
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

# Exogenous base features (should exist in loaded dataframe)
base_features = ['NO2(ppm)', 'U', 'V']

###############################################################################
def save_sarimax_formula_as_jpg(filename="formula_sarimax.jpg"):
    """
    Save SARIMAX model explanation (highlighting difference from SARIMA) as a JPEG image.
    Compatible with matplotlib mathtext.
    """
    # === Formula principale (solo parte matematica) ===
    formula_eq = (
        r"$y_t = c + \sum_{i=1}^{p}\phi_i y_{t-i}"
        r" + \sum_{I=1}^{P}\Phi_I y_{t-sI}"
        r" + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j}"
        r" + \sum_{J=1}^{Q}\Theta_J \varepsilon_{t-sJ}"
        r" + \beta^T X_t + \varepsilon_t$"
    )

    # === Testo esplicativo ===
    explanation_lines = [
        "SARIMAX model = SARIMA model extended with exogenous variables ($X_t$).",
        "",
        r"$y_t$: observed value at time $t$",
        r"$X_t$: exogenous variables (external predictors)",
        r"$\beta$: coefficients for $X_t$ (impact of external factors)",
        r"$\phi_i, \Phi_I$: autoregressive coefficients (non-seasonal, seasonal)",
        r"$\theta_j, \Theta_J$: moving-average coefficients (non-seasonal, seasonal)",
        r"$d, D$: differencing orders (non-seasonal, seasonal)",
        r"$s$: seasonal period (e.g., 24 for hourly data)",
        r"$\varepsilon_t$: white noise (random error)",
        "",
        "Goal: model both temporal dynamics and the effect of external regressors.",
    ]

    # === Creazione figura ===
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    # Titolo (solo testo normale)
    ax.text(0.5, 0.95, "SARIMAX Model Representation",
            fontsize=14, ha="center", va="top", fontweight="bold")

    # Formula centrata
    ax.text(0.5, 0.78, formula_eq, fontsize=14, ha="center", va="top")

    # Linee esplicative
    y_start = 0.55
    line_spacing = 0.06
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * line_spacing, line,
                fontsize=10, ha="left", va="top", wrap=True)

    plt.tight_layout()

    # === Salvataggio temporaneo ===
    temp_file = "_temp_formula_sarimax.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # === Conversione in JPEG ===
    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved SARIMAX formula image as {filename}")

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

# Ensure required exogenous columns exist
missing = [f for f in base_features if f not in df.columns]
if missing:
    raise RuntimeError(f"Missing required base features in dataframe: {missing}")

###############################################################################
# === Feature Engineering ===
# We follow the same feature engineering used for XGBoost to produce exogenous variables.
features = base_features.copy()

lag_features = {}
for l in range(1, 4):
    col_name = f'{target_item}_lag{l}'
    lag_features[col_name] = df[target_item].shift(l)
    features.append(col_name)

# concat lagged features
if lag_features:
    df = pd.concat([df, pd.DataFrame(lag_features, index=df.index)], axis=1)

rolling_features = {}
col_mean = f'{target_item}_roll_mean_3'
col_std = f'{target_item}_roll_std_6'
rolling_features[col_mean] = df[target_item].shift(1).rolling(3).mean()
rolling_features[col_std] = df[target_item].shift(1).rolling(6).std()
features += [col_mean, col_std]

if rolling_features:
    df = pd.concat([df, pd.DataFrame(rolling_features, index=df.index)], axis=1)

# Differences features
diff_features = {}
diff_features[f'{target_item}_diff_1'] = df[target_item].shift(1).diff(1)
features += list(diff_features.keys())

if diff_features:
    df = pd.concat([df, pd.DataFrame(diff_features, index=df.index)], axis=1)

# Cyclical hourly features and weekday flags
if "時" not in df.columns:
    raise RuntimeError("Column '時' (hour) is required in the dataframe for cyclical features.")

df["hour_sin"] = np.sin(2 * np.pi * df["時"] / 24)
df["dayofweek"] = df.index.dayofweek
features += ['hour_sin', 'dayofweek']

target_col = [target_item]

# Drop rows with NaNs in features or target
data_model = df.dropna(subset=features + target_col).copy()

###############################################################################
# === Prepare exogenous matrices and target series ===
X = data_model[features].copy()
y = data_model[target_col].squeeze()

# Scale exogenous variables to [0,1] to help optimization stability
scaler_X = MinMaxScaler()
X_scaled = pd.DataFrame(scaler_X.fit_transform(X), index=X.index, columns=X.columns)

# Train/test split (70/30)
split_index = int(len(X_scaled) * 0.7)
X_train, X_test = X_scaled.iloc[:split_index], X_scaled.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

###############################################################################
# === Non-seasonal grid (we search over p,d,q) ===
p_values = [1, 2]
d_values = [0]
q_values = [1, 2]

# Fixed seasonal params
P, D, Q, s = 1, 1, 1, 24

param_grid = {"p": p_values, "d": d_values, "q": q_values}

best_score = -np.inf
best_order = None
results = []

print("=== Starting SARIMAX grid search (non-seasonal parameters) ===")
for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                # Build SARIMAX with exogenous variables
                model = SARIMAX(
                    y_train,
                    exog=X_train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_fit = model.fit(disp=False)

                # Forecast using exogenous features for the test period
                y_pred = model_fit.forecast(steps=len(y_test), exog=X_test)

                # Evaluate
                r2 = r2_score(y_test, y_pred)
                results.append(((p, d, q), r2))
                print(f"Tested (p,d,q)=({p},{d},{q}) -> R2={r2:.6f}")
                if r2 > best_score:
                    best_score = r2
                    best_order = (p, d, q)
            except Exception as e:
                print(f"Failed (p,d,q)=({p},{d},{q}): {e}")
                continue

print("Best SARIMAX non-seasonal parameters found:", best_order)
print(f"Seasonal part fixed at: (P,D,Q,s)=({P},{D},{Q},{s})")

###############################################################################
# === Fit best SARIMAX model on full training data ===
best_model = SARIMAX(
    y_train,
    exog=X_train,
    order=best_order,
    seasonal_order=(P, D, Q, s),
    enforce_stationarity=False,
    enforce_invertibility=False
)
best_model_fit = best_model.fit(disp=False)

# Forecast on test set using exogenous variables
y_pred = best_model_fit.forecast(steps=len(y_test), exog=X_test)

###############################################################################
# === Evaluation metrics (out-of-sample) ===
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R²: {r2:.5f}")
print(f"MAE: {mae:.5f}")
print(f"RMSE: {rmse:.5f}")

###############################################################################
# === Residuals (out-of-sample diagnostics) ===
residuals = y_test.values - y_pred.values

# Compute statistics
res_mean = np.mean(residuals)
res_median = np.median(residuals)
# compute mode robustly across SciPy versions
try:
    res_mode = float(mode(residuals, nan_policy='omit').mode)
except Exception:
    # fallback: use the bin with highest histogram count as a proxy for mode
    hist_vals, bin_edges = np.histogram(residuals[~np.isnan(residuals)], bins=50)
    idx = np.argmax(hist_vals)
    res_mode = float((bin_edges[idx] + bin_edges[idx+1]) / 2)

# Ljung-Box test for autocorrelation in residuals
lb_test = acorr_ljungbox(residuals, lags=[1], return_df=True)
prob_Q = lb_test['lb_pvalue'].iloc[0] if not lb_test.empty else np.nan

# skewness and kurtosis
skewness = skew(residuals)
kurt = kurtosis(residuals, fisher=False)

###############################################################################
# === Save Figures for Report ===
figures = []

# 1) Predicted vs Real (last 720 samples)
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(y_test.values[-720:], label='Real values', color='gray')
ax1.plot(y_pred.values[-720:], label='SARIMAX forecast', color='lightgray', linestyle='dashed')
ax1.set_title(f'SARIMAX{best_order} x (P,D,Q,s)=({P},{D},{Q},{s})\nR²: {r2:.5f}')
ax1.set_xlabel('Samples')
ax1.set_ylabel(target_item)
ax1.legend()
ax1.grid(True)
fig1.tight_layout()
figures.append(fig1)

# 2) Residuals scatter
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.scatter(range(len(residuals)), residuals, color='gray', alpha=0.6, s=10)
ax2.axhline(0, color='gray', linestyle='--')
ax2.set_title('Residuals (Out-of-sample)')
ax2.set_xlabel('Samples')
ax2.set_ylabel('Residual (Real - Predicted)')
ax2.grid(True)
fig2.tight_layout()
figures.append(fig2)

# 3) Histogram of residuals with mean/median/mode lines
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
# Save formula image
formula_path = os.path.join('..', 'reports', 'formula_sarimax.jpg')
save_sarimax_formula_as_jpg(filename=formula_path)

###############################################################################
# === Parameters and Errors Summary ===
params_info = {
    'Prefecture code': prefecture_code,
    'Station code': station_code,
    'Station name': station_name,
    'Target item': target_item,
    'Number of training samples': len(y_train),
    'Number of testing samples': len(y_test),
    'Model': 'SARIMAX',
    'SARIMAX order (non-seasonal)': best_order,
    'Seasonal order (P,D,Q,s)': f'({P},{D},{Q},{s})',
}

# Add tested grid entries
#for key, values in param_grid.items():
#    params_info[f'Parameter Grid (tested) {key}'] = str(values)

# Add diagnostics and central tendencies
params_info['Predictions mean'] = np.mean(y_pred)
params_info['Predictions std'] = np.std(y_pred)
params_info['Real mean'] = np.mean(y_test.values)
params_info['Real std'] = np.std(y_test.values)
params_info['Ljung-Box residuals autocorrelation, Prob(Q)'] = prob_Q
params_info['Residuals skew'] = skewness
params_info['Residuals kurtosis'] = kurt

errors = [(target_item, r2, mae, rmse)]

###############################################################################
# === Save Report to PDF ===
report_file = f'sarimax_parameter_selection_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf'
report_path = os.path.join('..', 'reports', report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_report_to_pdf(
    filename=report_path,
    title=f"SARIMAX parameter selection Report - {station_name}",
    formula_image_path=formula_path,
    params=params_info,
    features=features,
    errors=errors,
    figures=figures
)

print(f"\n✅ Report saved as {report_path}")
