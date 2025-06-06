# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utility import load_and_prepare_data, WD_COLUMN, get_station_name
from report import save_report_to_pdf
import math
import numpy as np
from matplotlib.collections import PolyCollection

def customize_acf_plot(ax, font_size=8):
    if isinstance(ax, (list, np.ndarray)):
        for a in ax:
            customize_acf_plot(a, font_size)
        return

    ax.set_title(ax.get_title(), fontsize=font_size)
    ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)
    ax.set_ylabel(ax.get_ylabel(), fontsize=font_size)
    ax.tick_params(labelsize=font_size - 2)

    for line in ax.lines:
        line.set_linewidth(0.8)  # Minimum that has effect
        if hasattr(line, "set_markersize"):
            line.set_markersize(3)

    for item in ax.collections:
        if isinstance(item, PolyCollection):
            item.set_facecolor('darkgrey')

# === Parameters ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
station_name = get_station_name(data_dir, station_code)
max_lags = 48
rows_per_page = 6  # 1 feature per row => 6 features per page

# === Load data ===
data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)
items = [col for col in items if col != WD_COLUMN]

# === Plot configuration ===
plt.style.use('grayscale')
plots = []
items_per_page = rows_per_page  # 1 item per row

total_pages = math.ceil(len(items) / items_per_page)

for page_idx in range(total_pages):
    start_idx = page_idx * items_per_page
    end_idx = start_idx + items_per_page
    current_items = items[start_idx:end_idx]

    fig, axes = plt.subplots(
        nrows=rows_per_page,
        ncols=2,
        figsize=(8.3, 11.7),  # A4 portrait in inches
        constrained_layout=True
    )

    axes = axes.reshape(rows_per_page, 2)

    for row_idx, col in enumerate(current_items):
        ax_acf = axes[row_idx, 0]
        ax_pacf = axes[row_idx, 1]

        plot_acf(data[col], lags=max_lags, ax=ax_acf)
        ax_acf.set_title(f"ACF - {col}", fontsize=9)

        plot_pacf(data[col], lags=max_lags, ax=ax_pacf)
        ax_pacf.set_title(f"PACF - {col}", fontsize=9)

        customize_acf_plot([ax_acf, ax_pacf])

    # Turn off unused rows
    for j in range(len(current_items), rows_per_page):
        axes[j, 0].axis('off')
        axes[j, 1].axis('off')

    fig.suptitle(f'Autocorrelation Analysis — Page {page_idx + 1}', fontsize=12)
    plots.append(fig)

# === Save PDF ===
report_title = f'{station_name} - Autocorrelation Analysis'
report_file = f'Autocorrelation_{station_name}_{prefecture_code}_{station_code}.pdf'

model_params = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Max lags": max_lags,
    "Items per page": items_per_page,
    "Rows per page": rows_per_page,
    "Layout": "1 feature per row",
    "Color mode": "Grayscale",
}

errors = []

save_report_to_pdf(report_file, report_title, model_params, errors, plots)
