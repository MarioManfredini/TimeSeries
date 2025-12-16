# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 2025

@author: Mario
"""
# --------------------------------------------------------------
# Seasonal Decomposition Report Generator
# --------------------------------------------------------------
# This module performs a seasonal decomposition of a time series,
# visualizes trend, seasonal and residual components, and exports
# the results into a grayscale PDF report.
# --------------------------------------------------------------

import os
import matplotlib.pyplot as plt

from pathlib import Path
from statsmodels.tsa.seasonal import STL
from report import save_report_to_pdf
from PIL import Image
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf


from utility import load_and_prepare_data, get_station_name

###############################################################################
def save_seasonal_decomposition_formula_as_jpg(filename="formula_seasonal_decomposition.jpg"):
    """
    Save the Seasonal Decomposition formula (additive model) as a JPEG image with explanation.
    """
    # --- Formula (Additive model) ---
    formula = (
        r"$X_t = T_t + S_t + R_t$"
    )

    # --- Explanation lines ---
    explanation_lines = [
        r"$X_t$: observed value of the time series at time $t$",
        r"$T_t$: trend component (long-term progression of the series)",
        r"$S_t$: seasonal component (repeating short-term pattern)",
        r"$R_t$: residual component (random noise or irregular fluctuations)",
        "",
        r"In the additive model, the series is assumed to be the sum of these components.",
        r"For multiplicative models, the relationship becomes $X_t = T_t \times S_t \times R_t$.",
    ]

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")

    # --- Main formula ---
    ax.text(0, 1, formula, fontsize=22, ha="left", va="top")

    # --- Explanations ---
    y_start = 0.6
    line_spacing = 0.07
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * line_spacing, line,
                fontsize=11, ha="left", va="top")

    plt.tight_layout()

    # --- Save temporary PNG then convert to JPEG ---
    temp_file = "_temp_formula_seasonal.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved Seasonal Decomposition formula image as {filename}")

###############################################################################
# === Parameters ===
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'
period = 24 # hours

plt.style.use('grayscale')

def generate_decomposition_report():
    """
    Perform seasonal decomposition on a time series and export
    results into a PDF report with all subplots on one page.
    """
    
    try:
        # ----------------------------------------------------------
        # 1. Load time series data     
        # ----------------------------------------------------------
        data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)
        series = data[target_item].dropna()

        # ----------------------------------------------------------
        # 2. Perform seasonal decomposition
        # ----------------------------------------------------------
        stl = STL(series, period, robust=True)
        res = stl.fit()
        
        trend = res.trend
        seasonal = res.seasonal
        resid = res.resid

        # ----------------------------------------------------------
        # 3. Create a single figure with 5 stacked subplots
        # ----------------------------------------------------------
        fig, axes = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(11.7, 8.3),
            sharex=True
        )
        plt.subplots_adjust(hspace=0.5)
        
        # --- Original Series ---
        axes[0].plot(series, color='0.6', linewidth=0.5)
        axes[0].set_title(f"Original Time Series ({target_item} concentration)", fontsize=10)
        axes[0].set_ylabel(target_item)
        
        # --- Trend Component ---
        axes[1].plot(trend, color='0.2', linewidth=1.2)
        axes[1].set_title("Trend Component", fontsize=10)
        axes[1].set_ylabel("Trend")
        
        # --- Seasonal Component ---
        axes[2].plot(seasonal, color='0.7', linewidth=0.5)
        axes[2].set_title(f"Seasonal Component (Period = {period/24} days)", fontsize=10)
        axes[2].set_ylabel("Seasonality")
        
        # --- Residual Component ---
        axes[3].plot(resid, color='0.75', linewidth=0.5)
        axes[3].set_title("Residual Component", fontsize=10)
        axes[3].set_ylabel("Residuals")
        
        # --- Apply consistent grid ---
        for ax in axes:
            ax.grid(True, linestyle='--', linewidth=0.3, color='0.8')

        # ----------------------------------------------------------
        # 3b. Create ACF plot of residuals (new PDF page)
        # ----------------------------------------------------------
        resid_clean = resid.dropna()
        
        fig2, axes2 = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(11.7, 8.3),
            sharex=False
        )
        plt.subplots_adjust(hspace=0.5)
        
        # --- Q-Q Plot of Residuals ---
        resid_clean = resid.dropna()
        stats.probplot(
            resid_clean,
            dist="norm",
            plot=axes2[0]
        )
        qq_points = axes2[0].get_lines()[0]
        qq_points.set_marker('o')
        qq_points.set_markersize(2)  # marker size ~ sqrt(s)
        qq_points.set_color('0.6')
        norm_line = axes2[0].get_lines()[1]
        norm_line.set_color('0.8')
        axes2[0].set_title("Q–Q Plot of Residuals", fontsize=10)
        axes2[0].set_xlabel("Theoretical Quantiles")
        axes2[0].set_ylabel("Sample Quantiles")
        
        plot_acf(
            resid_clean,
            ax=axes2[1],
            lags=72,          # 3 days for hourly data
            alpha=0.05,
            zero=False
        )
        
        axes2[1].set_title("Autocorrelation Function (ACF) of Residuals", fontsize=12)
        axes2[1].set_xlabel("Lag (hours)")
        axes2[1].set_ylabel("Autocorrelation")
        
        axes2[1].grid(True, linestyle='--', linewidth=0.3, color='0.8')

        # Convert single figure into list for PDF
        figures = [fig, fig2]

        # ----------------------------------------------------------
        # 4. Prepare model parameters and text summary
        # ----------------------------------------------------------
        model_params = {
            "Prefecture code": prefecture_code,
            "Station code": station_code,
            "Station name": station_name,
            "Model type": "Additive decomposition",
            "Period": f"{period/24} days ({period} hours)",
            "Data points": len(series),
            "Start date": str(series.index.min()),
            "End date": str(series.index.max()),
        }
        
        # Save formula as image
        formula_path = os.path.join("..", "reports", "formula_seasonal_decomposition.jpg")
        save_seasonal_decomposition_formula_as_jpg(formula_path)
        
        explanation = (
                "The seasonal decomposition separates the time series into four components:\n"
                "1. **Original series** — the observed Ox concentration over time.\n"
                "2. **Trend component** — shows the long-term direction of variation.\n"
                "3. **Seasonal component** — highlights repeating periodic patterns, such as daily or monthly cycles.\n"
                "4. **Residual component** — contains random fluctuations not explained by the trend or seasonality.\n"
                "By examining these components, we can better understand whether variations in measurement levels\n"
                "are due to systematic seasonal patterns, long-term environmental changes, or random noise."
        )

        # ----------------------------------------------------------
        # 5. Save all into a PDF report
        # ----------------------------------------------------------
        report_file = f'seasonal_decomposition_stl_{station_name}_{prefecture_code}_{station_code}.pdf'
        report_path = os.path.join("..", "reports", report_file)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        save_report_to_pdf(
            filename=report_path,
            title=f"STL Decomposition Report - {station_name}",
            formula_image_path=formula_path,
            comments=explanation,
            params=model_params,
            figures=figures
        )

    except Exception as e:
        print(f"❌ Error generating decomposition report: {e}")

# --------------------------------------------------------------
# Run module directly (for standalone testing)
# --------------------------------------------------------------
if __name__ == "__main__":
    generate_decomposition_report()
