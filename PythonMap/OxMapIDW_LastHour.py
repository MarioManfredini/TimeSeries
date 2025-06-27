# -*- coding: utf-8 -*-
"""
Created: 2025/06/22

Author: Mario
"""
import os
import pandas as pd
import numpy as np
import folium
from folium.raster_layers import ImageOverlay
from map_utils import load_latest_ox_values, compute_wind_uv, compute_bounds_from_df, generate_idw_image, draw_arrow_wind_vectors, generate_idw_image_with_labels_only, generate_idw_loocv_error_image, generate_distance_mask, generate_ox_confidence_overlay_image
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial import cKDTree
from map_report import save_idw_report_pdf, capture_html_map_screenshot, save_idw_formula_as_jpg
import itertools

def evaluate_idw_loocv(df, k=5, power=1.0):
    coords = np.vstack((df['longitude'].values, df['latitude'].values)).T
    values = df['Ox(ppm)'].values

    preds = []
    trues = []

    for i in range(len(values)):
        x_train = np.delete(coords, i, axis=0)
        z_train = np.delete(values, i)

        point = coords[i].reshape(1, -1)

        tree = cKDTree(x_train)
        distances, idx = tree.query(point, k=k)
        weights = 1 / (distances + 1e-12) ** power
        z_pred = np.sum(weights * z_train[idx]) / np.sum(weights)

        preds.append(z_pred)
        trues.append(values[i])

    mse = mean_squared_error(trues, preds)
    rmse = mse ** 0.5
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    return rmse, mae, r2

# === CONFIG ===
data_dir = '..\\data\\Osaka\\'
prefecture_code = '27'
station_coordinates = 'Stations_Ox.csv'
csv_path = os.path.join(data_dir, station_coordinates)

# Load coordinates and Ox values
df = pd.read_csv(csv_path, skipinitialspace=True)

# === Load and prepare data ===
ox_data = load_latest_ox_values(data_dir, df, year=2025, month=5, prefecture_code=prefecture_code)
df = df[df['station_code'].isin(ox_data.keys())].copy()
df['Ox(ppm)'] = df['station_code'].map(lambda c: ox_data[c]['Ox(ppm)'])
df['WS(m/s)'] = df['station_code'].map(lambda c: ox_data[c]['WS(m/s)'])
df['WD(16Dir)'] = df['station_code'].map(lambda c: ox_data[c]['WD(16Dir)'])
df['U'], df['V'] = compute_wind_uv(df['WS(m/s)'], df['WD(16Dir)'])

# === test parameters ===
k_values = [5, 6, 7, 9]
power_values = [1.0, 1.2, 1.5, 2.0]

print("Grid search over IDW:")
print(" k  power |   RMSE   |   MAE   |   R²")
print("-" * 40)

results = []
for k, power in itertools.product(k_values, power_values):
    rmse, mae, r2 = evaluate_idw_loocv(df, k=k, power=power)
    print(f"{k:2d}  {power:5.2f} | {rmse:8.5f} | {mae:7.5f} | {r2:6.3f}")
    results.append((k, power, rmse, mae, r2))

results_sorted = sorted(results, key=lambda x: x[4], reverse=True)
print("\nBest parameters combination (max R²):")
for k, power, rmse, mae, r2 in results_sorted[:3]:
    print(f"k={k}, power={power} → RMSE={rmse:.5f}, MAE={mae:.5f}, R²={r2:.3f}")

k = results_sorted[0][0]
power = results_sorted[0][1]
print(f"Best: k={k}, power={power}")

# === Compute bounds and generate image ===
bounds = compute_bounds_from_df(df, margin_ratio=0.10)
# Color scale limits (percentiles)
ox_min = np.percentile(df['Ox(ppm)'], 5)
ox_max = np.percentile(df['Ox(ppm)'], 95)
ox_idw_image = 'ox_idw.png'

generate_ox_confidence_overlay_image(
    df,
    bounds,
    ox_min,
    ox_max,
    k,
    power,
    output_file=ox_idw_image,
    max_distance_km=15,
    cmap_name="Reds"
)

# === Folium Map ===
x = df['longitude'].values
y = df['latitude'].values
center_lat = y.mean()
center_lon = x.mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# === Overlay interpolated image ===
image_overlay = ImageOverlay(
    name="Interpolated Ox(ppm)",
    image=ox_idw_image,
    bounds=[[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]],
    opacity=0.7,
    interactive=False,
    cross_origin=False,
    zindex=1
)
image_overlay.add_to(m)

# Save interactive map
m.save("OxMapIDW_LastHour.html")
print("Map saved to: OxMapIDW_LastHour.html")

formula_image_path = 'formula_idw.jpg'
save_idw_formula_as_jpg(filename=formula_image_path)

idw_labels_image_path = 'ox_idw_with_labels_only.png'
generate_idw_image_with_labels_only(
    df, bounds, ox_min, ox_max, k, power, output_file='ox_idw_with_labels_only.png')

html_path = "OxMapIDW_LastHour.html"
screenshot_path = "map_screenshot.jpg"
capture_html_map_screenshot(html_path, screenshot_path)

generate_idw_loocv_error_image(
    df,
    bounds,
    k,
    power,
    output_file="error_map_rmse.png",
    metric='rmse'
)

save_idw_report_pdf(
    output_path="IDW_Report.pdf",
    results=results,
    formula_image_path=formula_image_path,
    html_screenshot_path=screenshot_path,
    idw_labels_image_path=idw_labels_image_path,
    title="IDW Cross-validation Report"
)
