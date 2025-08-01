# -*- coding: utf-8 -*-
"""
Created: 2025/07/06

Author: Mario
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium

from folium.raster_layers import ImageOverlay
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.patches import Rectangle
from pykrige.uk import UniversalKriging

from map_utils import (
    load_station_temporal_features,
    compute_bounds_from_df,
    generate_universal_kriging_grid,
    generate_confidence_overlay_image
)
from map_report import save_model_report_pdf, capture_html_map_screenshot, save_kriging_formula_as_jpg, plot_loocv_results

###############################################################################
def evaluate_universal_kriging_loocv(target, df, variogram_model='gaussian', transform=None, drift_terms=None):
    """
    LOOCV evaluation for Universal Kriging using external drift variables.

    Parameters:
        df: pandas DataFrame with 'longitude', 'latitude', measurements, and external drift variables
        variogram_model: str (e.g., 'gaussian', 'spherical')
        transform: 'log', 'sqrt', or None
        drift_terms: list of column names to use as external drift (e.g., ['NO(ppm)', 'NO2(ppm)', 'U', 'V'])

    Returns:
        rmse, mae, r2, trues, preds
    """
    epsilon = 1e-4
    x_all = df['longitude'].values
    y_all = df['latitude'].values
    z_raw = df[target].values

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

    if drift_terms is None or len(drift_terms) == 0:
        raise ValueError("drift_terms must be a non-empty list of external variables.")

    external_drift = df[drift_terms].values

    dups = df[['longitude', 'latitude']].duplicated().sum()
    print(f"[DEBUG] Number of duplicate coordinates: {dups}")

    preds = []
    trues = []

    for i in range(len(df)):
        try:
            x_train = np.delete(x_all, i)
            y_train = np.delete(y_all, i)
            z_train = np.delete(z_all, i)
            drift_train = np.delete(external_drift, i, axis=0)
            #print(f"x_train shape: {x_train.shape}, drift_train shape: {drift_train.shape}")
            
            x_test = x_all[i]
            y_test = y_all[i]

            uk = UniversalKriging(
                x_train, y_train, z_train,
                variogram_model=variogram_model,
                variogram_parameters=None,
                drift_terms=drift_terms,
                external_drift=drift_train,
                verbose=False,
                enable_plotting=False
            )

            z_pred_transf, _ = uk.execute('points', np.array([x_test]), np.array([y_test]))
            z_pred = inverse_transform(z_pred_transf[0])
            preds.append(z_pred)
            trues.append(z_raw[i])

        except Exception as e:
            print(f"⚠️ UK failed at point {i}: {e}")
            continue

    if len(trues) == 0:
        raise ValueError("❌ All Universal Kriging predictions failed.")

    mse = mean_squared_error(trues, preds)
    rmse = mse ** 0.5
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    return rmse, mae, r2, np.array(trues), np.array(preds)

###############################################################################
def generate_kriging_image_with_labels_only(
    df,
    bounds,
    vmin,
    vmax,
    variogram_model='gaussian',
    transform=None,
    output_file='kriging_labels.png',
    num_cells=800
):
    """
    Generate a Simple Kriging interpolated grayscale image with station locations and labels (no map, no wind).

    Parameters:
        df: DataFrame with columns ['longitude', 'latitude', measurements]
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        vmin, vmax: fixed grayscale scale
        variogram_model: variogram model type ('gaussian', 'spherical', etc.)
        output_file: path to save resulting PNG
        num_cells: resolution of interpolation grid
        transform: None, 'log', or 'sqrt' to apply to target variable
    """
    if vmin is None or vmax is None:
        raise ValueError("vmin and vmax must be provided to ensure consistent grayscale.")

    x = df['longitude'].values
    y = df['latitude'].values
    z_raw = df[target].values
    
    # === Apply transformation ===
    epsilon = 1e-4  # For log transform
    if transform == 'log':
        z = np.log(z_raw + epsilon)
        inverse_transform = lambda x: np.exp(x) - epsilon
    elif transform == 'sqrt':
        z = np.sqrt(z_raw)
        inverse_transform = lambda x: x ** 2
    else:
        z = z_raw
        inverse_transform = lambda x: x

    (lat_min, lon_min), (lat_max, lon_max) = bounds

    # === Create grid ===
    grid_x = np.linspace(lon_min, lon_max, num_cells)
    grid_y = np.linspace(lat_min, lat_max, num_cells)

    # === Fit Kriging model ===
    ok = OrdinaryKriging(
        x, y, z,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False
    )

    grid_z_transf, _ = ok.execute('grid', grid_x, grid_y)
    grid_z = inverse_transform(grid_z_transf)

    # === Plot ===
    aspect_ratio = (lat_max - lat_min) / (lon_max - lon_min)
    width = 6
    height = width * aspect_ratio

    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    ax.axis('off')

    ax.imshow(
        grid_z,
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin='lower',
        vmin=vmin,
        vmax=vmax,
        cmap='Greys',
        alpha=0.9,
        aspect='auto'
    )

    # Bounding box of measured area
    rect = Rectangle(
        (x.min(), y.min()),
        x.max() - x.min(),
        y.max() - y.min(),
        linewidth=1.5,
        edgecolor='black',
        linestyle='--',
        facecolor='none'
    )
    ax.add_patch(rect)

    # === Station markers and labels ===
    ax.scatter(x, y, c='black', edgecolor='white', s=50, zorder=5)

    for i, row in df.iterrows():
        ax.text(
            row['longitude'] + 0.01,
            row['latitude'] + 0.005,
            f"{row[target]:.3f}",
            fontsize=6,
            color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1),
            zorder=6
        )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_file, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Kriging label image saved to: {output_file}")

###############################################################################

# === CONFIG ===
data_dir = '..\\data\\Osaka\\'
prefecture_code = '27'
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
    lags=24,
    target_item=target
)

print("[DEBUG] Final dataframe shape:", df.shape)

drift_terms = df.columns.tolist()
print("[DEBUG] Columns:", drift_terms)

print("[DEBUG] Missing values per column:")
print(df.isna().sum())

dups = df[['longitude', 'latitude']].duplicated().sum()
if dups > 0:
    print(f"[WARN] Found {dups} duplicate coordinates.")

print("[DEBUG] External drift columns:", drift_terms)

external_drift = df[drift_terms]
print("[DEBUG] External drift sample values:")
print(external_drift.head())

print("[DEBUG] Missing values in external drift:")
print(external_drift.isna().sum())

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
            drift_terms=drift_terms
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
    drift_terms=drift_terms
)
loocv_image="loocv.png"
plot_loocv_results(target, rmse, mae, r2, trues, preds, loocv_image)

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
generate_confidence_overlay_image(
    df,
    bounds,
    grid,
    vmin,
    vmax,
    output_file,
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
    image=output_file,
    bounds=[[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]],
    opacity=0.7,
    interactive=False,
    cross_origin=False,
    zindex=1
)
image_overlay.add_to(m)

# Save interactive map
html_path = "map_one_hour_universal_kriging.html"
m.save(html_path)
print(f"✅ Map saved to: {html_path}")

formula_image_path = 'formula.jpg'
save_kriging_formula_as_jpg(formula_image_path)

kriging_labels_image_path = 'labels_image.png'
generate_kriging_image_with_labels_only(
    df,
    bounds,
    vmin,
    vmax,
    variogram_model=best_model,
    transform=best_transform,
    output_file=kriging_labels_image_path,
    num_cells=300
)

screenshot_path = "screenshot.jpg"
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

save_model_report_pdf(
    output_path="universal_kriging_report.pdf",
    table_data=table_data,
    column_headers=column_headers,
    formula_image_path=formula_image_path,
    map_image_path=screenshot_path,
    labels_image_path=kriging_labels_image_path,
    additional_image_path=loocv_image,
    title=f"Universal Kriging Interpolation - {year}/{month}/{day} {hour:02d}H"
)