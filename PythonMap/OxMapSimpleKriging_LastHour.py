# -*- coding: utf-8 -*-
"""
Created: 2025/06/29

Author: Mario
"""
import os
import pandas as pd
import numpy as np
import folium
from folium.raster_layers import ImageOverlay
from map_utils import (
    load_latest_ox_values,
    compute_wind_uv,
    compute_bounds_from_df,
    get_variogram_parameters,
    generate_simple_kriging_grid,
    generate_ox_kriging_confidence_overlay_image
)
from map_report import save_kriging_report_pdf, capture_html_map_screenshot, save_kriging_formula_as_jpg
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_kriging_loocv(df, variogram_model='linear', transform=None):
    """
    Evaluate LOOCV performance for Ordinary Kriging using a given variogram model,
    with optional transformation (e.g., 'log', 'sqrt').

    Parameters:
        df: DataFrame with 'longitude', 'latitude', 'Ox(ppm)'
        variogram_model: str, variogram model name
        transform: str or None. Options: 'log', 'sqrt', None

    Returns:
        rmse, mae, r2
    """
    epsilon = 1e-4  # For log transform

    x_all = df['longitude'].values
    y_all = df['latitude'].values
    z_raw = df['Ox(ppm)'].values

    # === Apply transformation ===
    if transform == 'log':
        z_all = np.log(z_raw + epsilon)
        inverse_transform = lambda x: np.exp(x) - epsilon
    elif transform == 'sqrt':
        z_all = np.sqrt(z_raw)
        inverse_transform = lambda x: x ** 2
    else:
        z_all = z_raw
        inverse_transform = lambda x: x

    preds = []
    trues = []

    for i in range(len(z_all)):
        x_train = np.delete(x_all, i)
        y_train = np.delete(y_all, i)
        z_train = np.delete(z_all, i)

        x_test = x_all[i]
        y_test = y_all[i]
        z_true = z_raw[i]  # always compare in original scale

        try:
            params = get_variogram_parameters(variogram_model)
            ok = OrdinaryKriging(
                x_train, y_train, z_train,
                variogram_model=variogram_model,
                variogram_parameters=params,
                verbose=False, enable_plotting=False
            )
            z_pred_transf, _ = ok.execute('points', np.array([x_test]), np.array([y_test]))
            z_pred = inverse_transform(z_pred_transf[0])
            preds.append(z_pred)
            trues.append(z_true)
        except Exception as e:
            print(f"⚠️ LOOCV failed at point {i}: {e}")
            continue

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

# === Load station coordinates ===
df = pd.read_csv(csv_path, skipinitialspace=True)

# === Load and merge latest Ox data ===
ox_data = load_latest_ox_values(data_dir, df, year=2025, month=5, prefecture_code=prefecture_code)
df = df[df['station_code'].isin(ox_data.keys())].copy()
df['Ox(ppm)'] = df['station_code'].map(lambda c: ox_data[c]['Ox(ppm)'])
df['WS(m/s)'] = df['station_code'].map(lambda c: ox_data[c]['WS(m/s)'])
df['WD(16Dir)'] = df['station_code'].map(lambda c: ox_data[c]['WD(16Dir)'])
df['U'], df['V'] = compute_wind_uv(df['WS(m/s)'], df['WD(16Dir)'])

models = ['linear', 'gaussian', 'exponential', 'spherical']
transforms = [None, 'log', 'sqrt']

print("Variogram  | Transform |   RMSE   |   MAE    |   R²")
print("-" * 50)
results = []

for model in models:
    for transform in transforms:
        rmse, mae, r2 = evaluate_kriging_loocv(df, variogram_model=model, transform=transform)
        t_label = transform if transform else "none"
        print(f"{model:10s} | {t_label:9s} | {rmse:8.5f} | {mae:8.5f} | {r2:6.3f}")
        results.append((model, t_label, rmse, mae, r2))

best = max(results, key=lambda x: x[4])
print(f"\n✅ Best model by R²: {best[0]}, {best[1]} transform → RMSE={best[2]:.5f}, MAE={best[3]:.5f}, R²={best[4]:.3f}")

# === Compute bounds and color scale ===
bounds = compute_bounds_from_df(df, margin_ratio=0.10)
ox_min = np.percentile(df['Ox(ppm)'], 5)
ox_max = np.percentile(df['Ox(ppm)'], 95)

# === Generate Kriging interpolated image ===
kriging_image_path = 'ox_kriging.png'
ox_grid, (lat_cells, lon_cells), grid_x, grid_y = generate_simple_kriging_grid(
    df,
    bounds,
    num_cells=300,
    variogram_model='spherical' # 'linear', 'gaussian', 'exponential', 'spherical'
)
generate_ox_kriging_confidence_overlay_image(
    df=df,
    bounds=bounds,
    ox_grid=ox_grid,  # interpolazione Kriging già calcolata
    ox_min=ox_min,
    ox_max=ox_max,
    output_file="ox_kriging.png",
    max_distance_km=15,
    cmap_name="Reds"
)

# === Create folium map with image overlay ===
x = df['longitude'].values
y = df['latitude'].values
center_lat = y.mean()
center_lon = x.mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

image_overlay = ImageOverlay(
    name="Simple Kriging Ox(ppm)",
    image=kriging_image_path,
    bounds=[[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]],
    opacity=0.7,
    interactive=False,
    cross_origin=False,
    zindex=1
)
image_overlay.add_to(m)

# Save interactive map
m.save("OxMapSimpleKriging_LastHour.html")
print("✅ Map saved to: OxMapSimpleKriging_LastHour.html")

formula_image_path = 'formula_kriging.jpg'
save_kriging_formula_as_jpg(formula_image_path)

html_path = "OxMapSimpleKriging_LastHour.html"
screenshot_path = "map_screenshot.jpg"
capture_html_map_screenshot(html_path, screenshot_path)

save_kriging_report_pdf(
    output_path="kriging_report.pdf",
    results=results,  # lista di tuple (model, transform, rmse, mae, r2)
    formula_image_path=formula_image_path,
    html_screenshot_path=screenshot_path,
    kriging_labels_image_path="ox_kriging_with_labels_only.png"
)
