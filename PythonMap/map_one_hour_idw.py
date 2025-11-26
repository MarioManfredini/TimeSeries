# -*- coding: utf-8 -*-
"""
Created: 2025/06/22

Author: Mario
"""
import os
import numpy as np
import itertools
import folium

from pathlib import Path
from folium.raster_layers import ImageOverlay

from map_utils import (
    load_preprocessed_hourly_data,
    compute_bounds_from_df,
    kriging_interpolate_residuals,
    generate_idw_grid,
    generate_confidence_overlay_image
)
from map_report import save_model_report_pdf, capture_html_map_screenshot
from map_one_hour_idw_helper import (
    idw_loocv,
    save_rf_formula_as_jpg,
    generate_idw_image_with_labels_only
)

###############################################################################
# === CONFIG ===
data_dir = Path('..') / 'data' / 'Osaka'
prefecture_code = '27'
prefecture_name = '大阪府'
station_coordinates = 'Stations_Ox.csv'
target = 'Ox(ppm)'
year = 2025
month = 5
day = 12
hour = 19
num_cells = 300

###############################################################################
# === Load data ===
df = load_preprocessed_hourly_data(
    data_dir,
    station_coordinates,
    prefecture_code,
    year,
    month,
    day,
    hour,
    target
)

# === test parameters ===
k_values = [8, 9, 10]
power_values = [0.05, 0.1, 0.7, 0.8, 0.9, 1.0]

print("Grid search over IDW:")
print(" k  power |   RMSE   |   MAE   |   R²")
print("-" * 40)

results = []
for k, power in itertools.product(k_values, power_values):
    rmse, mae, r2, _ = idw_loocv(target, df, k, power)
    print(f"{k:2d}  {power:5.2f} | {rmse:8.5f} | {mae:7.5f} | {r2:6.3f}")
    results.append((k, power, rmse, mae, r2))

results_sorted = sorted(results, key=lambda x: x[4], reverse=True)
print("\nBest parameters combination (max R²):")
for k, power, rmse, mae, r2 in results_sorted[:3]:
    print(f"k={k}, power={power} → RMSE={rmse:.5f}, MAE={mae:.5f}, R²={r2:.3f}")

k = results_sorted[0][0]
power = results_sorted[0][1]
print(f"Best: k={k}, power={power}")

loocv_image="loocv.png"
loocv_image_path = os.path.join(".", "tmp", loocv_image)
os.makedirs(os.path.dirname(loocv_image_path), exist_ok=True)

rmse, mae, r2, residuals_df = idw_loocv(target, df, k, power, loocv_image)

# === Compute bounds ===
bounds = compute_bounds_from_df(df, margin_ratio=0.10)

interpolated_residuals_image = "kriging_of_xgb_residuals.png"
interpolated_residuals_image_path = os.path.join(".", "tmp", interpolated_residuals_image)
os.makedirs(os.path.dirname(interpolated_residuals_image_path), exist_ok=True)
kriging_interpolate_residuals(residuals_df, bounds, output_file=interpolated_residuals_image_path)

# === Generate IDW interpolated grid ===
grid = generate_idw_grid(
    target,
    df,
    bounds,
    num_cells,
    k,
    power
)

# === Generate IDW confidence overlay image ===
output_file = 'prediction.png'
output_file_path = os.path.join(".", "tmp", output_file)
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Color scale limits (percentiles)
vmin = np.percentile(df[target], 5)
vmax = np.percentile(df[target], 95)
generate_confidence_overlay_image(
    df,
    bounds,
    grid,
    vmin,
    vmax,
    output_file,
    cmap_name='Reds'
)

# === Folium Map ===
x = df['longitude'].values
y = df['latitude'].values
center_lat = y.mean()
center_lon = x.mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# === Overlay interpolated image ===
image_overlay = ImageOverlay(
    name=f"Interpolated {target}",
    image=output_file,
    bounds=[[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]],
    opacity=0.7,
    interactive=False,
    cross_origin=False,
    zindex=1
)
image_overlay.add_to(m)

###############################################################################
# Save map
html_file = (
    f'map_idw_one_hour_{prefecture_name}_{target}_{year}{month}{day}{hour}.html'
)
html_path = os.path.join("..", "html", html_file)
os.makedirs(os.path.dirname(html_path), exist_ok=True)
m.save(html_path)
print(f"Map saved to: {html_path}")

formula_image = 'formula.jpg'
formula_image_path = os.path.join(".", "tmp", formula_image)
os.makedirs(os.path.dirname(formula_image_path), exist_ok=True)
save_rf_formula_as_jpg(filename=formula_image_path)

idw_labels_image = 'prediction.png'
idw_labels_image_path = os.path.join(".", "tmp", idw_labels_image)
os.makedirs(os.path.dirname(idw_labels_image_path), exist_ok=True)
generate_idw_image_with_labels_only(
    target,
    df,
    bounds,
    grid,
    vmin,
    vmax,
    k,
    power,
    output_file=idw_labels_image_path
)

screenshot = "screenshot.jpg"
screenshot_path = os.path.join(".", "tmp", screenshot)
os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
capture_html_map_screenshot(html_path, screenshot_path)

# === parameters test results ===
column_headers = ["k", "power", "RMSE", "MAE", "R²"]
table_data = []
for k, power, rmse, mae, r2 in results_sorted:
    table_data.append([
        f"{k}",
        f"{power:.2f}",
        f"{rmse:.5f}",
        f"{mae:.5f}",
        f"{r2:.3f}"
    ])

###############################################################################
# === Save Report ===
report_file = (
    f'map_idw_{prefecture_name}_{target}_{year}{month:02}{day:02}_{hour:02}00.pdf'
)
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_model_report_pdf(
    output_path=report_path,
    table_data=table_data,
    column_headers=column_headers,
    formula_image_path=formula_image_path,
    map_image_path=screenshot_path,
    labels_image_path=idw_labels_image_path,
    additional_image_path=loocv_image,
    title=f"IDW - {prefecture_name} - {year}/{month}/{day} {hour:02d}H"
)
