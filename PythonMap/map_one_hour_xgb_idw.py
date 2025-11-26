# -*- coding: utf-8 -*-
"""
Created: 2025/11/24
Author: Mario
"""
import os
import pandas as pd
import numpy as np
import folium

from pathlib import Path
from folium.raster_layers import ImageOverlay
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from map_utils import (
    load_station_temporal_features,
    compute_bounds_from_df,
    kriging_interpolate_residuals,
    generate_combined_confidence_overlay_image,
    generate_model_idw_labels_image
)
from map_report import (
    capture_html_map_screenshot,
    save_model_report_pdf,
    plot_loocv_results
)
from map_one_hour_xgb_idw_helper import (
    compute_xgb_loocv_residuals,
    save_xgb_idw_formula_as_jpg
)

###############################################################################
# === CONFIG ===
data_dir = Path('..') / 'data' / 'Osaka'
prefecture_code = '27'
prefecture_name = '大阪府'
station_coordinates = 'Stations_Ox.csv'
csv_path = os.path.join(data_dir, station_coordinates)
target = 'Ox(ppm)'
year = 2025
month = 5
day = 12
hour = 19

###############################################################################
# === Load and prepare data ===
df = pd.read_csv(csv_path, skipinitialspace=True)

df = load_station_temporal_features(
    data_dir,
    df,
    prefecture_code,
    year,
    month,
    day,
    hour,
    lags=24,
    target_item=target
)

print("[DEBUG] Final dataframe shape:", df.shape)

# Exclude non-numeric columns from features list
exclude_cols = ['datetime', 'WD(16Dir)', 'station_code', 'station_name']
features = [col for col in df.columns if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)]

###############################################################################
# === Compute residuals using XGBoost model (LOOCV) ===
xgb_model, residuals_df, scaler = compute_xgb_loocv_residuals(df, features, target)

mse = mean_squared_error(residuals_df[target], residuals_df['prediction'])
rmse = mse ** 0.5
mae = mean_absolute_error(residuals_df[target], residuals_df['prediction'])
r2 = r2_score(residuals_df[target], residuals_df['prediction'])

print("\n✅ XGBoost LOOCV:")
print(f"RMSE: {rmse:.5f}, MAE: {mae:.5f}, R²: {r2:.3f}")

###############################################################################
# === LOOCV plot ===
loocv_image = "xgb_loocv.png"
loocv_image_path = os.path.join(".", "tmp", loocv_image)
os.makedirs(os.path.dirname(loocv_image_path), exist_ok=True)
plot_loocv_results(
    target,
    rmse,
    mae,
    r2,
    residuals_df[target].values,
    residuals_df['prediction'].values,
    loocv_image_path
)

# === Generate kriging map of residuals ===
bounds = compute_bounds_from_df(df, margin_ratio=0.10)

interpolated_residuals_image = "kriging_of_xgb_residuals.png"
interpolated_residuals_image_path = os.path.join(".", "tmp", interpolated_residuals_image)
os.makedirs(os.path.dirname(interpolated_residuals_image_path), exist_ok=True)
kriging_interpolate_residuals(residuals_df, bounds, output_file=interpolated_residuals_image_path)

# === Combined XGB + IDW image (prediction overlay) ===
vmin = np.percentile(df[target], 5)
vmax = np.percentile(df[target], 95)

output_file = 'xgb_idw_prediction.png'
output_file_path = os.path.join(".", "tmp", output_file)
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

generate_combined_confidence_overlay_image(
    df=df,
    model=xgb_model,
    scaler=scaler,
    features=features,
    bounds=bounds,
    target=target,
    vmin=vmin,
    vmax=vmax,
    output_file=output_file_path,
    num_cells=300,
    max_distance_km=15,
    cmap_name="Reds",
    p=2  # IDW power
)

# === Folium map ===
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
m_combined = folium.Map(location=[center_lat, center_lon], zoom_start=10)

ImageOverlay(
    name="XGBoost + IDW",
    image=output_file_path,
    bounds=[[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]],
    opacity=0.7
).add_to(m_combined)

html_file = f'map_xgb_idw_one_hour_{prefecture_name}_{target}_{year}{month}{day}{hour}.html'
html_path = os.path.join("..", "html", html_file)
os.makedirs(os.path.dirname(html_path), exist_ok=True)
m_combined.save(html_path)
print(f"✅ Map saved to: {html_path}")

# === Formula image ===
formula_image_path = os.path.join(".", "tmp", "formula_xgb_idw.jpg")
os.makedirs(os.path.dirname(formula_image_path), exist_ok=True)
save_xgb_idw_formula_as_jpg(formula_image_path)

# === Labels image (predicted values at station locations) ===
labels_image = 'xgb_labels_image.png'
labels_image_path = os.path.join(".", "tmp", labels_image)
os.makedirs(os.path.dirname(labels_image_path), exist_ok=True)
generate_model_idw_labels_image(
    df=df,
    bounds=bounds,
    target=target,
    vmin=vmin,
    vmax=vmax,
    model=xgb_model,
    scaler=scaler,
    features=features,
    output_file=labels_image_path,
    num_cells=500,
    p=2
)

# === Screenshot HTML ===
screenshot = "screenshot_xgb.jpg"
screenshot_path = os.path.join(".", "tmp", screenshot)
os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
capture_html_map_screenshot(html_path, screenshot_path)

# === Report PDF ===
results = [(rmse, mae, r2)]
column_headers = ["RMSE", "MAE", "R²"]
table_data = [[f"{rmse:.5f}", f"{mae:.5f}", f"{r2:.3f}"] for (rmse, mae, r2) in results]

report_file = f'map_xgb_idw_{prefecture_name}_{target}_{year}{month:02}{day:02}_{hour:02}00.pdf'
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_model_report_pdf(
    output_path=report_path,
    table_data=table_data,
    column_headers=column_headers,
    formula_image_path=formula_image_path,
    map_image_path=screenshot_path,
    labels_image_path=labels_image_path,
    additional_image_path=loocv_image_path,
    residuals_kriging_image_path=interpolated_residuals_image_path,
    title=f"XGBoost Interpolation and IDW - {prefecture_name} - {year}/{month}/{day} {hour:02d}H"
)
