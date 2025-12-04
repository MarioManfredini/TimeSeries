# -*- coding: utf-8 -*-
"""
Created: 2025/07/06

Author: Mario
"""
import os
import pandas as pd
import numpy as np
import folium

from pathlib import Path
from folium.raster_layers import ImageOverlay
from statsmodels.stats.outliers_influence import variance_inflation_factor

from map_utils import (
    load_station_temporal_features,
    add_utm_coordinates,
    compute_bounds_from_df,
    generate_confidence_overlay_image
)
from map_report import (
    save_model_report_pdf,
    capture_html_map_screenshot,
    save_kriging_formula_as_jpg,
    plot_loocv_results
)
from map_one_hour_universal_kriging_helper import (
    evaluate_universal_kriging_loocv,
    generate_universal_kriging_image_with_labels_only,
    generate_universal_kriging_grid
)

###############################################################################
# === CONFIG ===
data_dir = Path('..') / 'data' / 'Osaka'
prefecture_code = '27'
prefecture_name = '大阪府'
station_coordinates = 'Stations_Ox.csv'
csv_path = os.path.join(data_dir, station_coordinates)
target = 'Ox(ppm)'
year=2025
month=5
day=12
hour=19

# === Load station coordinates ===
df = pd.read_csv(csv_path, skipinitialspace=True)

df = load_station_temporal_features(
    data_dir,
    df,
    prefecture_code,
    year,
    month,
    day,
    hour,
    lags=1,
    target_item=target
)

print("[DEBUG] Final dataframe shape:", df.shape)

# The Universal Kriging model works best with UTM coordinates
# although it is not possible to specify the coordinate type
# when calling the pykrige.uk UniversalKriging function.
df = add_utm_coordinates(df)

# === Select only numeric drift terms ===
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove coordinates and target from drift
# Universal Kriging uses a built-in linear regression on the drifts. If some drifts are highly collinear:
# - the coefficients become unstable
# - small changes in the data generate huge changes in the weights
# - linear systems become difficult to invert
exclude = [
    'station_code', 'longitude', 'latitude', 'x_m', 'y_m', target,
    # features
    'NO2(ppm)',
    # lags
    'Ox(ppm)_lag1', 'NO2(ppm)_lag1',
    # diffs
    'Ox(ppm)_diff_1', 'Ox(ppm)_diff_2', 'Ox(ppm)_diff__3',
    'NO(ppm)_diff_3', 'NO2(ppm)_diff_3', 'U_diff_3', 'V_diff_3',
    # rolling mean/std
    'Ox(ppm)_roll_mean_3', 'Ox(ppm)_roll_std_6',
    'NO(ppm)_roll_mean_3', 'NO(ppm)_roll_std_6',
    'NO2(ppm)_roll_mean_3', 'NO2(ppm)_roll_std_6',
    'U_roll_mean_3', 'U_roll_std_6',
    'V_roll_mean_3', 'V_roll_std_6',
    # hour
    'hour_sin', 'hour_cos'
]
drift_terms = [c for c in numeric_cols if c not in exclude]

X = df[drift_terms].fillna(0).values
vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print("Variance Inflation Factor (VIF):")
# VIFᵢ = 1 / (1 − R²ᵢ)
# where R²ᵢ is obtained by regressing Xᵢ on all other features.
for col, vif in zip(drift_terms, vifs):
    print(col, vif)

# basic checks
print("[DEBUG] Drift terms used (numeric only):")
print(drift_terms)
print("[DEBUG] Missing values per column:")
print(df.isna().sum())
dups = df[['longitude', 'latitude']].duplicated().sum()
if dups > 0:
    print(f"[WARN] Found {dups} duplicate coordinates.")
print("[DEBUG] External drift columns:", drift_terms)
print("[DEBUG] n stations:", len(df))
print("[DEBUG] Any NaN in target?", df[target].isna().any())
print("[DEBUG] Any NaN in drift columns?", df[drift_terms].isna().any().any())
print("[DEBUG] duplicate coords:", df[['longitude','latitude']].duplicated().sum())
print("[DEBUG] drift shape:", df[drift_terms].shape)

models = ['linear', 'gaussian', 'exponential', 'spherical']
transforms = [None, 'log', 'sqrt']

print("Variogram  | Transform |   RMSE   |   MAE    |   R²")
print("-" * 50)
results = []
for model in models:
    for transform in transforms:
        rmse, mae, r2, trues, preds = evaluate_universal_kriging_loocv(
            target,
            df,
            variogram_model=model,
            transform=transform,
            drift_terms=drift_terms,
            x_col='x_m',
            y_col='y_m',
            variogram_parameters=None
        )
        t_label = transform if transform else "none"
        print(f"{model:10s} | {t_label:9s} | {rmse:8.5f} | {mae:8.5f} | {r2:6.3f}")
        results.append((model, t_label, rmse, mae, r2))

best = max(results, key=lambda x: x[4])
best_model = best[0]
best_transform = best[1]
print(f"\n✅ Best model by R²: {best_model}, {best_transform} transform → RMSE={best[2]:.5f}, MAE={best[3]:.5f}, R²={best[4]:.3f}")

rmse, mae, r2, trues, preds = evaluate_universal_kriging_loocv(
    target,
    df,
    variogram_model=best_model,
    transform=best_transform,
    drift_terms=drift_terms,
    x_col='x_m',
    y_col='y_m',
    variogram_parameters=None
    )
loocv_image="loocv.png"
loocv_image_path = os.path.join(".", "tmp", loocv_image)
os.makedirs(os.path.dirname(loocv_image_path), exist_ok=True)
plot_loocv_results(target, rmse, mae, r2, trues, preds, loocv_image_path)

# === Compute bounds and color scale ===
bounds = compute_bounds_from_df(df, margin_ratio=0.10)
vmin = np.percentile(df[target], 5)
vmax = np.percentile(df[target], 95)

# === Generate Kriging interpolated image ===
grid = generate_universal_kriging_grid(
    target,
    df,
    bounds,
    drift_terms,
    num_cells=300,
    variogram_model=best_model
)

# === Generate confidence overlay image ===
output_file = 'prediction.png'
output_file_path = os.path.join(".", "tmp", output_file)
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
generate_confidence_overlay_image(
    df,
    bounds,
    grid,
    vmin,
    vmax,
    output_file_path,
    cmap_name='Reds'
)

# === Create folium map with image overlay ===
x = df['longitude'].values
y = df['latitude'].values
center_lat = y.mean()
center_lon = x.mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

image_overlay = ImageOverlay(
    name=f"Universal Kriging {target}",
    image=output_file_path,
    bounds=[[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]],
    opacity=0.7,
    interactive=False,
    cross_origin=False,
    zindex=1
)
image_overlay.add_to(m)

# Save interactive map
html_file = (
    f'map_universal_kriging_one_hour_{prefecture_name}_{target}_{year}{month}{day}{hour}.html'
)
html_path = os.path.join("..", "html", html_file)
os.makedirs(os.path.dirname(html_path), exist_ok=True)
m.save(html_path)
print(f"✅ Map saved to: {html_path}")

formula_image = 'formula.jpg'
formula_image_path = os.path.join(".", "tmp", formula_image)
os.makedirs(os.path.dirname(formula_image_path), exist_ok=True)
save_kriging_formula_as_jpg(formula_image_path)

labels_image = 'labels_image.png'
labels_image_path = os.path.join(".", "tmp", labels_image)
os.makedirs(os.path.dirname(labels_image_path), exist_ok=True)
generate_universal_kriging_image_with_labels_only(
    target,
    df,
    bounds,
    vmin,
    vmax,
    drift_terms=drift_terms,
    variogram_model=best_model,
    transform=best_transform,
    output_file=labels_image_path,
    num_cells=300
)

screenshot = "screenshot.jpg"
screenshot_path = os.path.join(".", "tmp", screenshot)
os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
capture_html_map_screenshot(html_path, screenshot_path)

column_headers = ["Variogram", "Transform", "RMSE", "MAE", "R²"]
table_data = []
for model, t_label, rmse, mae, r2 in results:
    table_data.append([
        f"{model}",
        f"{t_label}",
        f"{rmse:.5f}",
        f"{mae:.5f}",
        f"{r2:.3f}"
    ])

###############################################################################
# === Save Report ===
report_file = (
    f'map_universal_kriging_{prefecture_name}_{target}_{year}{month:02}{day:02}_{hour:02}00.pdf'
)
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
    title=f"Universal Kriging Interpolation - {prefecture_name} - {year}/{month}/{day} {hour:02d}H"
)
