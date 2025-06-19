# -*- coding: utf-8 -*-
"""
Created 2025/06/04
@modified: Mario
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from windrose import WindroseAxes
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utility import load_and_prepare_data, get_station_name
from sklearn.preprocessing import MinMaxScaler

target_item = 'Ox(ppm)'

# === Load data ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

# === Feature Engineering ===
lags = 24
features = ['NO(ppm)', 'NO2(ppm)', 'U', 'V']
lagged_items = [target_item, 'NO(ppm)', 'NO2(ppm)', 'U', 'V']

lag_features = {}
for item in lagged_items:
    for l in range(1, lags):
        col_name = f'{item}_lag{l}'
        lag_features[col_name] = data[item].shift(l)
        features.append(col_name)

data = pd.concat([data, pd.DataFrame(lag_features, index=data.index)], axis=1)

rolling_features = {}
for item in lagged_items:
    col_mean = f'{item}_roll_mean_3'
    col_std = f'{item}_roll_std_6'
    rolling_features[col_mean] = data[item].shift(1).rolling(3).mean()
    rolling_features[col_std] = data[item].shift(1).rolling(6).std()
    features += [col_mean, col_std]

data = pd.concat([data, pd.DataFrame(rolling_features, index=data.index)], axis=1)

diff_features = {}
diff_features[f'{target_item}_diff_1'] = data[target_item].shift(1).diff(1)
diff_features[f'{target_item}_diff_2'] = data[target_item].shift(1).diff(2)
diff_features[f'{target_item}_diff_3'] = data[target_item].shift(1).diff(3)
features += list(diff_features.keys())

for item in ['NO(ppm)', 'NO2(ppm)', 'U', 'V']:
    col_name = f'{item}_diff_3'
    diff_features[col_name] = data[item].shift(1).diff(3)
    features.append(col_name)

data = pd.concat([data, pd.DataFrame(diff_features, index=data.index)], axis=1)

data["hour_sin"] = np.sin(2 * np.pi * data["時"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["時"] / 24)
data["dayofweek"] = data.index.dayofweek
data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)
features += ['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend']

data_model = data.dropna(subset=features).copy()

# === FEATURE NORMALIZATION ===
scaler = MinMaxScaler()
data_model[features] = scaler.fit_transform(data_model[features])

# === -1 shifted TARGET (one-step-ahead) ===
data_model[target_item] = data_model[target_item].shift(-1)
data_model = data_model.dropna()

# === TRAIN/TEST ===
y = data_model[target_item]
X = data_model[features]
split_index = int(len(y) * 0.7)
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]

# === Fit SARIMAX (simple params) ===
model = SARIMAX(y_train,
                exog=X_train,
                order=(1, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit(disp=False)

# === Predict one-step-ahead ===
y_pred = results.predict(start=split_index, end=len(y)-1, exog=X_test)
residuals = y_test - y_pred

# === Metrics ===
r2_val = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# === Create A4 report ===
fig = plt.figure(figsize=(8.27, 11.69))
gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1.5, 0.5])
plt.rcParams['font.family'] = 'Yu Gothic'

# === Metrics Summary ===
ax_metrix = fig.add_subplot(gs[0, 0])
ax_metrix.text(0, 0.99, f'{station_name} - Ox Prediction Analysis', fontsize=16, fontweight='bold', ha='left', va='top')
ax_metrix.axis('off')
text = f"""
Prefecture code: {prefecture_code}
Station code: {station_code}
Station name: {station_name}
Model: SARIMAX(1,0,0) with exogenous features

Target item: {target_item}

R²: {r2_val:.5f}
MAE: {mae:.5f}
MSE: {mse:.5f}
RMSE: {rmse:.5f}
Mean of {target_item}: {np.mean(y_test):.5f}
Test Samples: {len(y_test)}
"""
ax_metrix.text(0, 0.80, text, va='top', ha='left', fontsize=8)

# === Windrose ===
pos = gs[0, 1].get_position(fig)
ax_windrose = WindroseAxes(fig, [
    pos.x0 + pos.width * 0.2,
    pos.y0 + pos.height * 0.5,
    pos.width,
    pos.height,
])
fig.add_axes(ax_windrose)
wind_data = data[['WD_degrees', 'WS(m/s)', target_item]].dropna()
ax_windrose.bar(wind_data['WD_degrees'], wind_data[target_item],
                normed=True, opening=0.8, edgecolor='white',
                colors=['#cccccc', '#aaaaaa', '#999999', '#666666', '#333333', '#000000'],
                bins=np.linspace(wind_data[target_item].min(), wind_data[target_item].max(), 6))
for label in ax_windrose.get_xticklabels():
    label.set_fontsize(6)
for label in ax_windrose.get_yticklabels():
    label.set_fontsize(5)
legend = ax_windrose.set_legend(title=target_item, decimal_places=3, fontsize=5, title_fontsize=6)
legend.get_frame().set_linewidth(0.5)
legend.set_bbox_to_anchor((-0.8, 0.5))
legend._loc = 6
legend.set_frame_on(False)

# === Heatmap ===
ax_heatmap = fig.add_subplot(gs[1, 0])
wind_data['Dir_bin'] = pd.cut(wind_data['WD_degrees'], bins=np.arange(0, 361, 22.5), right=False)
wind_data['Speed_bin'] = pd.cut(wind_data['WS(m/s)'], bins=np.arange(0, wind_data['WS(m/s)'].max() + 1, 1), right=False)
heatmap_data = wind_data.pivot_table(index='Dir_bin', columns='Speed_bin',
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

# === Scatter U/V ===
ax_scatter = fig.add_subplot(gs[1, 1])
scatter = ax_scatter.scatter(data['U'], data['V'], c=data[target_item], cmap='Greys', alpha=0.7)
cbar = fig.colorbar(scatter, ax=ax_scatter)
ax_scatter.set_xlabel('U (m/s)', fontsize=6)
ax_scatter.set_ylabel('V (m/s)', fontsize=6)
ax_scatter.set_title(f'{target_item} by Wind Components', fontsize=8)
ax_scatter.grid(True)
ax_scatter.tick_params(labelsize=6)
cbar.ax.tick_params(labelsize=6)

# === Predicted vs Actual ===
from matplotlib.gridspec import GridSpecFromSubplotSpec
gs_sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, :], height_ratios=[3, 1], hspace=0.05)
ax_pred = fig.add_subplot(gs_sub[0])
y_true_i = y_test.values[:168]
y_pred_i = y_pred.values[:168]
residuals_i = y_true_i - y_pred_i
std_res = np.std(residuals_i)

ax_pred.plot(y_true_i, label='Actual', color='gray')
ax_pred.plot(y_pred_i, label='Predicted', color='black', linestyle='--', linewidth=0.8)
ax_pred.fill_between(np.arange(len(y_pred_i)),
                     y_pred_i - std_res,
                     y_pred_i + std_res,
                     color='lightgray', alpha=0.6,
                     label=f'±{std_res:.3f}')
ax_pred.set_title(f'Predicted vs Actual - R²: {r2_val:.5f}', fontsize=8)
ax_pred.set_ylabel(target_item, fontsize=6)
ax_pred.tick_params(axis='y', labelsize=5)
ax_pred.legend(fontsize=5)
ax_pred.set_xticks([])

# === Residuals ===
ax_resid = fig.add_subplot(gs_sub[1], sharex=ax_pred)
ax_resid.plot(residuals_i, color='gray', linewidth=0.7)
ax_resid.axhline(std_res, color='gray', linestyle='--', linewidth=0.5)
ax_resid.axhline(-std_res, color='gray', linestyle='--', linewidth=0.5)
ax_resid.set_ylabel('Residuals', fontsize=6)
ax_resid.tick_params(axis='both', labelsize=5)

# Highlight out-of-band
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

# === Histogram of residuals ===
ax_histogram = fig.add_subplot(gs[3, :])
sns.histplot(residuals_i, kde=True, bins=30, ax=ax_histogram, color='gray', edgecolor='black', alpha=0.7)
ax_histogram.set_title('Residual Error Histogram', fontsize=8)
ax_histogram.set_xlabel('Residuals', fontsize=6)
ax_histogram.set_ylabel('Count', fontsize=6)
ax_histogram.tick_params(axis='both', labelsize=5)
ax_histogram.grid(True, linewidth=0.3, alpha=0.5)

# === Save to PDF ===
pdf_filename = f'SARIMAX_report_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf'
with PdfPages(pdf_filename) as pdf:
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print("📄 One-page SARIMAX summary saved")
