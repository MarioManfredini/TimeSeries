# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from windrose import WindroseAxes
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utility import load_and_prepare_data, get_station_name
from report import set_japanese_font

###############################################################################
# === Parameters ===
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Feature Engineering ===
features = ['NO(ppm)', 'NO2(ppm)', 'U', 'V']
engineered_items = [target_item, 'NO(ppm)', 'NO2(ppm)', 'U', 'V']

# Rolling mean and std features
rolling_features = {}
for item in engineered_items:
    col_mean = f'{item}_roll_mean_3'
    col_std = f'{item}_roll_std_6'
    rolling_features[col_mean] = df[item].shift(1).rolling(3).mean()
    rolling_features[col_std] = df[item].shift(1).rolling(6).std()
    features += [col_mean, col_std]

df = pd.concat([df, pd.DataFrame(rolling_features, index=df.index)], axis=1)

# Difference features
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

# Time-related features
df["hour_sin"] = np.sin(2 * np.pi * df["時"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["時"] / 24)
df["dayofweek"] = df.index.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
features += ['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend']

target_col = [target_item]

# Drop missing rows
data_model = df.dropna(subset=features + target_col).copy()

###############################################################################
# === Train/Test Split (80/20) ===
n_samples = len(data_model)
train_size = int(0.8 * n_samples)

X_train_raw = data_model[features].iloc[:train_size]
y_train_raw = data_model[target_col].iloc[:train_size]
X_val_raw = data_model[features].iloc[train_size:]
y_val_raw = data_model[target_col].iloc[train_size:]

###############################################################################
# === Normalization (fit only on training data) ===
scaler_X = MinMaxScaler().fit(X_train_raw)
scaler_y = MinMaxScaler().fit(y_train_raw)

X_train = pd.DataFrame(scaler_X.transform(X_train_raw), columns=features)
X_val = pd.DataFrame(scaler_X.transform(X_val_raw), columns=features)
y_train = pd.DataFrame(scaler_y.transform(y_train_raw), columns=target_col)
y_val = pd.DataFrame(scaler_y.transform(y_val_raw), columns=target_col)

###############################################################################
# === Model trainig and validation ===
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
residuals = y_val.values - y_pred

r2_val = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)

###############################################################################
# === Save Report to PDF ===
import os

report_file = f'linear_regression_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf'
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

# === Create A4 PDF with 5x2 layout ===
fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1.5, 0.5])  # 5 rows x 2 columns

# Font
font_name = set_japanese_font()
plt.rcParams['font.family'] = font_name

###############################################################################
# === Metrics Summary ===
ax_metrix = fig.add_subplot(gs[0, 0])
ax_metrix.text(0, 0.99, f'{station_name} - オキシダント予測の分析', fontsize=16, fontweight='bold', ha='left', va='top')
ax_metrix.axis('off')
text = f"""
Prefecture code: {prefecture_code}
Station code: {station_code}
Station name: {station_name}
Model: Linear Regression (no lag)

Target item: {target_item}

R²: {r2_val:.5f}
MAE: {mae:.5f}
MSE: {mse:.5f}
RMSE: {rmse:.5f}
Mean of {target_item}: {np.average(df[target_item]):.5f}
Test Samples: {len(y_val)}
"""
ax_metrix.text(0, 0.80, text, va='top', ha='left', fontsize=8)

###############################################################################
# === Formula ===
ax_formula = fig.add_subplot(gs[0, 1])
ax_formula.axis("off")

formula_eq = (
    r"$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p$" "\n"
    r"$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$"
)

explanation_lines = [
    r"$\hat{y}$: predicted value",
    r"$x_j$: j-th input feature (e.g., NO, NO₂, wind components, ...)",
    r"$\beta_j$: model coefficient (learned weight)",
    r"$\beta_0$: intercept term (bias)",
    r"$\mathcal{L}$: mean squared error minimized during training",
]

y_pos = 0.8
ax_formula.text(0.0, y_pos, "Linear Regression Model", fontsize=10,
                fontweight='bold', ha="left", va="top")
ax_formula.text(0.0, y_pos - 0.15, formula_eq, fontsize=9,
                ha="left", va="top", linespacing=1.4)

y_start = y_pos - 0.50
for i, line in enumerate(explanation_lines):
    ax_formula.text(0.0, y_start - i * 0.06, line, fontsize=7,
                    ha="left", va="top")

###############################################################################
# === Heatmap ===
ax_heatmap = fig.add_subplot(gs[1, 0])
wind_and_item = df[['WD_degrees', 'WS(m/s)', target_item]].dropna()
wind_and_item['Dir_bin'] = pd.cut(wind_and_item['WD_degrees'], bins=np.arange(0, 361, 22.5), right=False)
wind_and_item['Speed_bin'] = pd.cut(wind_and_item['WS(m/s)'], bins=np.arange(0, wind_and_item['WS(m/s)'].max() + 1, 1), right=False)
heatmap_data = wind_and_item.pivot_table(index='Dir_bin', columns='Speed_bin',
                                         values=target_item, aggfunc='mean', observed=False)
im = ax_heatmap.imshow(heatmap_data, aspect='auto', cmap='Greys', origin='lower')
cbar = fig.colorbar(im, ax=ax_heatmap, label=target_item)
ax_heatmap.set_xticks(np.arange(len(heatmap_data.columns)))
ax_heatmap.set_xticklabels([f"{x.left:.1f}" for x in heatmap_data.columns], rotation=90, fontsize=6)
ax_heatmap.set_yticks(np.arange(len(heatmap_data.index)))
ax_heatmap.set_yticklabels([f"{x.left:.1f}" for x in heatmap_data.index], fontsize=6)
ax_heatmap.set_title('Heatmap: Wind Direction/Speed vs Ox(ppm)', fontsize=8)
ax_heatmap.set_xlabel('Wind Speed (m/s)', fontsize=6)
ax_heatmap.set_ylabel('Wind Direction (°)', fontsize=6)
cbar.ax.tick_params(labelsize=6)
cbar.set_label(target_item, fontsize=6)

###############################################################################
# === Windrose ===
pos = gs[1, 1].get_position(fig)
centered_rect = [
    pos.x0 + pos.width * 0.3,
    pos.y0 + pos.height * 0.35,
    pos.width,
    pos.height,
]
ax_windrose = WindroseAxes(fig, centered_rect)
fig.add_axes(ax_windrose)
wind_and_item = df[['WD_degrees', 'WS(m/s)', target_item]].dropna()
ax_windrose.bar(wind_and_item['WD_degrees'], wind_and_item[target_item],
        normed=True, opening=0.8, edgecolor='white',
        colors=['#cccccc', '#aaaaaa', '#999999', '#666666', '#333333', '#000000'],
        bins=np.linspace(wind_and_item[target_item].min(), wind_and_item[target_item].max(), 6))
for label in ax_windrose.get_xticklabels():
    label.set_fontsize(6)
for label in ax_windrose.get_yticklabels():
    label.set_fontsize(5)
legend = ax_windrose.set_legend(title=target_item, decimal_places=3, fontsize=4, title_fontsize=6)
legend.get_frame().set_linewidth(0.5)
legend.set_bbox_to_anchor((-0.65, 0.5))
legend._loc = 6  # 'center left'
legend.set_frame_on(False)

###############################################################################
# === Predicted vs Actual ===
gs_sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, :], height_ratios=[3, 1], hspace=0.05)
ax_pred = fig.add_subplot(gs_sub[0])
y_true_i = y_val.values[:168].ravel()
y_pred_i = y_pred[:168].ravel()
residuals = y_true_i - y_pred_i
std_res = np.std(residuals)

ax_pred.plot(y_true_i, label='Actual', color='gray')
ax_pred.plot(y_pred_i, label='Predicted', color='black', linestyle='--', linewidth=0.8)
ax_pred.fill_between(np.arange(len(y_pred_i)),
                     y_pred_i - std_res,
                     y_pred_i + std_res,
                     color='lightgray', alpha=0.6,
                     label=f'±{std_res:.3f}')
ax_pred.set_title(f'Predicted vs Actual - R²: {r2_val:.5f}', fontsize=8)
ax_pred.set_xticks([])
ax_pred.set_ylabel(target_item, fontsize=6)
ax_pred.tick_params(axis='y', labelsize=5, colors='black')
ax_pred.legend(fontsize=5)

###############################################################################
# === Residuals ===
ax_resid = fig.add_subplot(gs_sub[1], sharex=ax_pred)
ax_resid.plot(residuals, color='gray', linewidth=0.7)
ax_resid.axhline(std_res, color='gray', linestyle='--', linewidth=0.5)
ax_resid.axhline(-std_res, color='gray', linestyle='--', linewidth=0.5)
ax_resid.set_ylabel('Residuals', fontsize=6)
ax_resid.tick_params(axis='both', labelsize=5, colors='gray')
# ± std
out_of_band = (y_true_i < (y_pred_i - std_res)) | (y_true_i > (y_pred_i + std_res))
in_interval = False
start_idx = None
for j, out in enumerate(out_of_band):
    if out and not in_interval:
        in_interval = True
        start_idx = j
    elif not out and in_interval:
        in_interval = False
        ax_resid.axvspan(start_idx, j, color='darkgrey', alpha=0.3)
if in_interval:
    ax_resid.axvspan(start_idx, len(out_of_band), color='darkgrey', alpha=0.3)
ax_resid.set_xlabel('Samples', fontsize=6)

###############################################################################
# === Histogram of residuals ===
ax_histogram = fig.add_subplot(gs[3, :])
sns.histplot(residuals, kde=True, bins=30, ax=ax_histogram, color='gray', edgecolor='black', alpha=0.7)
ax_histogram.set_title('Residual Error Histogram', fontsize=8)
ax_histogram.set_xlabel('Residuals', fontsize=6)
ax_histogram.set_ylabel('Count', fontsize=6)
ax_histogram.tick_params(axis='both', labelsize=5)
ax_histogram.grid(True, linewidth=0.3, alpha=0.5)

# === Save to PDF ===
with PdfPages(report_path) as pdf:
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"\n✅ Report saved as {report_path}")
