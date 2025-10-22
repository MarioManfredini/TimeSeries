# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utility import load_and_prepare_data, WD_COLUMN, get_station_name
from report import save_report_to_pdf
import math
import numpy as np
from matplotlib.collections import PolyCollection
from PIL import Image

###############################################################################
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

###############################################################################
def save_acf_pacf_formula_as_jpg(filename="formula_acf_pacf.jpg"):
    """
    Save ACF/PACF formula as a JPEG image with explanation.
    """
    formula = (
        r"$\rho(k) = \frac{\sum_{t=k+1}^{T}(x_t - \bar{x})(x_{t-k} - \bar{x})}{\sum_{t=1}^{T}(x_t - \bar{x})^2}$"
    )

    explanation_lines = [
        r"$\rho(k)$: autocorrelation at lag $k$",
        r"$x_t$: value of the series at time $t$",
        r"$\bar{x}$: mean of the time series",
        r"$T$: total number of observations",
        r"For PACF, the correlation is adjusted for intermediate lags.",
    ]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")

    ax.text(0, 1, formula, fontsize=20, ha="left", va="top")

    y_start = 0.4
    line_spacing = 0.07
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * line_spacing, line,
                fontsize=12, ha="left", va="top")

    plt.tight_layout()

    temp_file = "_temp_formula_acf_pacf.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved ACF/PACF formula image as {filename}")

###############################################################################
# === Parameters ===
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38206050'
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
        figsize=(8.3, 11.7),
    )
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08, wspace=0.25, hspace=0.35)

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

    fig.suptitle('Autocorrelation Analysis', fontsize=12)
    plots.append(fig)

# === Save PDF ===
report_title = f'Autocorrelation Analysis - {station_name}'
report_file = f'autocorrelation_{station_name}_{prefecture_code}_{station_code}.pdf'

report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

# Save formula as image
formula_path = os.path.join("..", "reports", "formula_acf_pacf.jpg")
save_acf_pacf_formula_as_jpg(formula_path)

explanation = (
        "The Autocorrelation Function (ACF) measures how the current values of a time series are correlated with\n"
        "its past values (lags). A high ACF at lag k means that values k steps apart are similar.\n\n"
        "The Partial Autocorrelation Function (PACF) isolates the direct correlation between a value and its k-th lag,\n"
        "removing the effect of intermediate lags.\n\n"
        "Interpreting the plots:\n"
        "- A slow decay in ACF indicates non-stationarity (trend or seasonality).\n"
        "- A sharp cutoff in PACF suggests an AR(p) structure.\n"
        "- A sharp cutoff in ACF suggests an MA(q) structure."
)

params = {
    "Prefecture code": prefecture_code,
    "Station code": station_code,
    "Station name": station_name,
    "Max lags": max_lags,
    "Data points": len(data),
    "Start date": str(data.index.min()),
    "End date": str(data.index.max()),
}

save_report_to_pdf(
    report_path,
    report_title,
    formula_image_path=formula_path,
    comments=explanation,
    params=params,
    figures=plots,
)
