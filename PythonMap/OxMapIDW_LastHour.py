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
from map_utils import (
                       load_hour_ox_values,
                       compute_wind_uv,
                       compute_bounds_from_df,
                       generate_idw_loocv_error_image,
                       generate_ox_confidence_overlay_image
                       )
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial import cKDTree
from map_report import save_idw_report_pdf, capture_html_map_screenshot, save_idw_formula_as_jpg
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

###############################################################################
def generate_idw_image_with_labels_only(df, bounds, ox_min, ox_max, k=7, power=1, output_file='ox_idw_labels.png', num_cells=800):
    """
    Generate an IDW interpolated grayscale image with station locations and Ox labels (no map, no wind).

    Parameters:
        df: DataFrame with columns ['longitude', 'latitude', 'Ox(ppm)']
        bounds: tuple ((lat_min, lon_min), (lat_max, lon_max))
        output_file: path to save resulting PNG
        num_cells: resolution of interpolation grid
        ox_min, ox_max: fixed min/max values for grayscale scale
    """
    if ox_min is None or ox_max is None:
        raise ValueError("ox_min and ox_max must be provided to ensure consistent grayscale.")

    x = df['longitude'].values
    y = df['latitude'].values
    z = df['Ox(ppm)'].values

    (lat_min, lon_min), (lat_max, lon_max) = bounds

    # === Create interpolation grid ===
    grid_x, grid_y = np.meshgrid(
        np.linspace(lon_min, lon_max, num_cells),
        np.linspace(lat_min, lat_max, num_cells)
    )
    grid_coords = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    # === IDW interpolation ===
    tree = cKDTree(np.vstack((x, y)).T)
    distances, idx = tree.query(grid_coords, k=k)
    weights = 1 / (distances + 1e-12) ** power
    values = np.sum(weights * z[idx], axis=1) / np.sum(weights, axis=1)
    grid_z = values.reshape((num_cells, num_cells))

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
        vmin=ox_min,
        vmax=ox_max,
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
            f"{row['Ox(ppm)']:.3f}",
            fontsize=6,
            color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1),
            zorder=6
        )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_file, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ IDW label image saved to: {output_file}")

###############################################################################
def plot_idw_loocv_and_save(df, k=7, power=1.0, output_file="idw_loocv_results.jpg"):
    coords = np.vstack((df['longitude'].values, df['latitude'].values)).T
    trues = df['Ox(ppm)'].values

    y_pred = []
    y_true = []

    for i in range(len(df)):
        x_train = np.delete(coords, i, axis=0)
        y_train = np.delete(trues, i)
        x_test = coords[i].reshape(1, -1)

        tree = cKDTree(x_train)
        distances, idx = tree.query(x_test, k=k)

        distances = np.maximum(distances, 1e-12)
        weights = 1 / (distances ** power)
        z_pred = np.sum(weights * y_train[idx]) / np.sum(weights)

        y_pred.append(z_pred)
        y_true.append(trues[i])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Metriche
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    # --- Scatter plot
    axs[0].scatter(y_true, y_pred, alpha=0.8)
    axs[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    axs[0].set_xlabel("True Ox(ppm)")
    axs[0].set_ylabel("Predicted Ox(ppm)")
    axs[0].set_title(f"IDW LOOCV - True vs Predicted\nRMSE={rmse:.5f}, MAE={mae:.5f}, R²={r2:.3f}")
    axs[0].grid(True)
    axs[0].axis("equal")

    # --- Line plot
    axs[1].plot(y_true, label="True", color="black", linewidth=1.5)
    axs[1].plot(y_pred, label="Predicted", color="blue", linestyle="--")
    axs[1].set_title("True vs Predicted (Index order)")
    axs[1].set_xlabel("Index")
    axs[1].set_ylabel("Ox(ppm)")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"✅ IDW LOOCV completed. Image saved to: {output_file}")
    print(f"RMSE: {rmse:.5f}, MAE: {mae:.5f}, R²: {r2:.3f}")
    return rmse, mae, r2

###############################################################################

# === CONFIG ===
data_dir = '..\\data\\Osaka\\'
prefecture_code = '27'
station_coordinates = 'Stations_Ox.csv'
csv_path = os.path.join(data_dir, station_coordinates)

# Load coordinates and Ox values
df = pd.read_csv(csv_path, skipinitialspace=True)

# === Load and prepare data ===
ox_data = load_hour_ox_values(
    data_dir,
    df,
    prefecture_code=prefecture_code,
    year=2025,
    month=5,
    day=12,
    hour=12
)
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
ox_idw_loocv="ox_idw_loocv.png"
for k, power in itertools.product(k_values, power_values):
    rmse, mae, r2 = plot_idw_loocv_and_save(df, k, power, ox_idw_loocv)
    print(f"{k:2d}  {power:5.2f} | {rmse:8.5f} | {mae:7.5f} | {r2:6.3f}")
    results.append((k, power, rmse, mae, r2))

results_sorted = sorted(results, key=lambda x: x[4], reverse=True)
print("\nBest parameters combination (max R²):")
for k, power, rmse, mae, r2 in results_sorted[:3]:
    print(f"k={k}, power={power} → RMSE={rmse:.5f}, MAE={mae:.5f}, R²={r2:.3f}")

k = results_sorted[0][0]
power = results_sorted[0][1]
print(f"Best: k={k}, power={power}")

rmse, mae, r2 = plot_idw_loocv_and_save(df, k, power, ox_idw_loocv)

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
    df, bounds, ox_min, ox_max, k, power, output_file=idw_labels_image_path)

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
    additional_image_path=ox_idw_loocv,
    title="IDW Cross-validation Report"
)
