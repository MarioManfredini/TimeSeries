# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from windrose import WindroseAxes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utility import load_and_prepare_data, get_station_name
import seaborn as sns
from matplotlib.gridspec import GridSpecFromSubplotSpec

target_item = 'Ox(ppm)'

# === Load data ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

# === Prepare data ===
features = ['SO2(ppm)', 'NO(ppm)', 'NO2(ppm)',
           'NMHC(ppmC)', 'CH4(ppmC)', 'SPM(mg/m3)', 'PM2.5(μg/m3)',
           'U', 'V']

X = data[features]
y = data[target_item]
split_index = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
residuals = y_test.values - y_pred

r2_val = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# === Create A4 PDF with 5x2 layout ===
fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1.5, 0.5])  # 5 rows x 2 columns

plt.rcParams['font.family'] = 'Yu Gothic'

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
Mean of {target_item}: {np.average(data[target_item]):.5f}
Test Samples: {len(y_test)}
"""
ax_metrix.text(0, 0.80, text, va='top', ha='left', fontsize=8)

# === Windrose ===
pos = gs[0, 1].get_position(fig)
centered_rect = [
    pos.x0 + pos.width * 0.2,
    pos.y0 + pos.height * 0.5,
    pos.width,
    pos.height,
]
ax_windrose = WindroseAxes(fig, centered_rect)
fig.add_axes(ax_windrose)
wind_and_item = data[['WD_degrees', 'WS(m/s)', target_item]].dropna()
ax_windrose.bar(wind_and_item['WD_degrees'], wind_and_item[target_item],
        normed=True, opening=0.8, edgecolor='white',
        colors=['#cccccc', '#aaaaaa', '#999999', '#666666', '#333333', '#000000'],
        bins=np.linspace(wind_and_item[target_item].min(), wind_and_item[target_item].max(), 6))
for label in ax_windrose.get_xticklabels():
    label.set_fontsize(6)
for label in ax_windrose.get_yticklabels():
    label.set_fontsize(5)
legend = ax_windrose.set_legend(title=target_item, decimal_places=3, fontsize=5, title_fontsize=6)
legend.get_frame().set_linewidth(0.5)
legend.set_bbox_to_anchor((-0.8, 0.5))
legend._loc = 6  # 'center left'
legend.set_frame_on(False)

# === Heatmap ===
ax_heatmap = fig.add_subplot(gs[1, 0])
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

# === Scatter U/V ===
ax_scatter = fig.add_subplot(gs[1, 1])
scatter = ax_scatter.scatter(data['U'], data['V'], c=data[target_item], cmap='Greys', alpha=0.7)
cbar = fig.colorbar(scatter, ax=ax_scatter)
cbar.set_label(target_item)
ax_scatter.set_xlabel('U (m/s)', fontsize=6)
ax_scatter.set_ylabel('V (m/s)', fontsize=6)
ax_scatter.set_title(f'{target_item} by Wind Components', fontsize=8)
ax_scatter.grid(True)
ax_scatter.tick_params(labelsize=6)
cbar.ax.tick_params(labelsize=6)
cbar.set_label(target_item, fontsize=6)

# === Predicted vs Actual ===
gs_sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, :], height_ratios=[3, 1], hspace=0.05)
ax_pred = fig.add_subplot(gs_sub[0])
y_true_i = y_test.values[:168]
y_pred_i = y_pred[:168]
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


# === Histogram of residuals ===
ax_histogram = fig.add_subplot(gs[3, :])
sns.histplot(residuals, kde=True, bins=30, ax=ax_histogram, color='gray', edgecolor='black', alpha=0.7)
ax_histogram.set_title('Residual Error Histogram', fontsize=8, color='black')
ax_histogram.set_xlabel('Residuals', fontsize=6, color='black')
ax_histogram.set_ylabel('Count', fontsize=6, color='black')
ax_histogram.tick_params(axis='both', labelsize=5, colors='black')
ax_histogram.grid(True, linewidth=0.3, alpha=0.5)

# === Save to PDF ===
with PdfPages(f'LinearRegression_report_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf') as pdf:
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print("📄 One-page summary saved")
