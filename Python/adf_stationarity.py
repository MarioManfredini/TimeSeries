# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 2025

@author: Mario
"""
# --------------------------------------------------------------
# ADF Stationarity Test Report Generator
# --------------------------------------------------------------
# This module applies the Augmented Dickey-Fuller (ADF) test to a
# time series (original and differenced versions), visualizes them,
# and exports results into a grayscale PDF report.
# --------------------------------------------------------------

import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from report import save_report_to_pdf
from PIL import Image

from utility import load_and_prepare_data, get_station_name

###############################################################################
# === Parameters ===
data_dir = Path('..') / 'data' / 'Ehime'
prefecture_code = '38'
station_code = '38206050'
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'

plt.style.use('grayscale')

###############################################################################
def save_adf_formula_as_jpg(filename="formula_adf.jpg"):
    """
    Save the Augmented Dickey-Fuller (ADF) test formula as a JPEG image with explanation.
    """
    # --- Main formula ---
    formula = (
        r"$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + "
        r"\sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t$"
    )

    # --- Explanation lines ---
    explanation_lines = [
        r"$\Delta y_t$: first difference of the series ($y_t - y_{t-1}$)",
        r"$\alpha$: constant term (drift)",
        r"$\beta t$: deterministic trend component",
        r"$\gamma$: coefficient testing the presence of a unit root",
        r"$\delta_i$: coefficients of lagged differences",
        r"$\varepsilon_t$: white noise error term",
        "",
        r"Null hypothesis $H_0$: the series has a unit root (non-stationary)",
        r"Alternative hypothesis $H_1$: the series is stationary",
    ]

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.axis("off")

    # --- Draw formula ---
    ax.text(0, 1, formula, fontsize=20, ha="left", va="top")

    # --- Draw explanations ---
    y_start = 0.6
    line_spacing = 0.07
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * line_spacing, line,
                fontsize=11, ha="left", va="top")

    plt.tight_layout()

    # --- Save temporary PNG then convert to JPG ---
    temp_file = "_temp_formula_adf.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved ADF formula image as {filename}")

###############################################################################
def perform_adf_test(series: pd.Series) -> dict:
    """Run the Augmented Dickey-Fuller test and return results as a dictionary."""
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Used Lags": result[2],
        "Number of Observations": result[3],
        "Critical Values": result[4],
    }

###############################################################################
def generate_adf_report():
    """
    Perform ADF test on original and differenced series and export
    results into a PDF report with visual comparison.
    """
    try:
        # ----------------------------------------------------------
        # 1. Load time series data
        # ----------------------------------------------------------
        data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)
        series = data[target_item].dropna()

        # ----------------------------------------------------------
        # 2. Perform ADF test
        # ----------------------------------------------------------
        adf_original = perform_adf_test(series)
        diff_series = series.diff().dropna()
        adf_diff = perform_adf_test(diff_series)

        # ----------------------------------------------------------
        # 3. Create figure with two subplots (original and differenced)
        # ----------------------------------------------------------
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(11.7, 8.3),
            sharex=True,
        )
        plt.subplots_adjust(hspace=0.4)

        # --- Original Series ---
        axes[0].plot(series, color='0.5', linewidth=0.6)
        axes[0].set_title(f"Original Time Series ({target_item})", fontsize=10)
        axes[0].set_ylabel(target_item)
        axes[0].grid(True, linestyle='--', linewidth=0.3, color='0.85')

        # --- Differenced Series ---
        axes[1].plot(diff_series, color='0.3', linewidth=0.6)
        axes[1].set_title("Differenced Series (1st order)", fontsize=10)
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Δ Value")
        axes[1].grid(True, linestyle='--', linewidth=0.3, color='0.85')

        figures = [fig]

        # ----------------------------------------------------------
        # 4. Prepare model parameters and ADF results summary
        # ----------------------------------------------------------
        model_params = {
            "Prefecture code": prefecture_code,
            "Station code": station_code,
            "Station name": station_name,
            "Target item": target_item,
            "ADF model": "Augmented Dickey-Fuller Test",
            "Series length": len(series),
            "Start date": str(series.index.min()),
            "End date": str(series.index.max()),

            "Original Series ADF Statistic": adf_original["ADF Statistic"],
            "Original Series p-value": adf_original["p-value"],
            "Original Series Used Lags": adf_original["Used Lags"],

            "Differenced Series ADF Statistic": adf_diff["ADF Statistic"],
            "Differenced Series p-value": adf_diff["p-value"],
            "Differenced Series Used Lags": adf_diff["Used Lags"],
        }
        
        # Save formula as image
        formula_path = os.path.join("..", "reports", "formula_adf.jpg")
        save_adf_formula_as_jpg(formula_path)
        
        explanation = (
            "The Augmented Dickey-Fuller (ADF) test is used to check whether a time series is stationary — that is,\n"
            "whether its statistical properties such as mean and variance remain constant over time.\n"
            "Mathematically, the test is based on an autoregressive (AR) model.\n"
            "The decision is based on the p-value:\n"
            "• A small p-value (typically < 0.05) suggests strong evidence against H₀, meaning the series is stationary.\n"
            "• A large p-value indicates that H₀ cannot be rejected — the series is likely non-stationary.\n"
            "In simple terms, the p-value measures how likely it is to observe the given data if the null hypothesis were true.\n"
            "Lower values mean the data are unlikely under H₀, strengthening the evidence that the series is stationary."
        )

        # ----------------------------------------------------------
        # 5. Save report as PDF
        # ----------------------------------------------------------
        report_file = f"adf_stationarity_{station_name}_{prefecture_code}_{station_code}.pdf"
        report_path = os.path.join("..", "reports", report_file)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        save_report_to_pdf(
            filename=report_path,
            title=f"ADF Stationarity Report - {station_name}",
            formula_image_path=formula_path,
            comments=explanation,
            params=model_params,
            figures=figures
        )

        print(f"✅ ADF report saved to: {report_path}")

    except Exception as e:
        print(f"❌ Error generating ADF report: {e}")


# --------------------------------------------------------------
# Run module directly (for standalone testing)
# --------------------------------------------------------------
if __name__ == "__main__":
    generate_adf_report()
