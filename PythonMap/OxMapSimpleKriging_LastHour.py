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
    load_hour_ox_values,
    compute_wind_uv,
    compute_bounds_from_df,
    get_variogram_parameters,
    generate_simple_kriging_grid,
    generate_ox_kriging_confidence_overlay_image
)
from map_report import save_kriging_report_pdf, capture_html_map_screenshot, save_kriging_formula_as_jpg
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


###############################################################################
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
    return rmse, mae, r2, trues, preds

###############################################################################
def plot_kriging_loocv_results(rmse, mae, r2, trues, preds, output_path="kriging_loocv_plot.jpg"):
    """
    Generates and saves a combined plot:
    - Top: scatter plot (true vs predicted)
    - Bottom: line plot (true and predicted values)

    Parameters:
        trues: list or array of true Ox values
        preds: list or array of predicted Ox values
        output_path: path to save the resulting image
    """
    trues = np.array(trues)
    preds = np.array(preds)

    fig, axs = plt.subplots(2, 1, figsize=(6, 8), dpi=200)

    # === 1. Scatter plot ===
    axs[0].scatter(trues, preds, alpha=0.8)
    axs[0].plot([trues.min(), trues.max()], [trues.min(), trues.max()], 'r--')
    axs[0].set_xlabel("True Ox(ppm)")
    axs[0].set_ylabel("Predicted Ox(ppm)")
    axs[0].set_title(f"Kriging LOOCV - True vs Predicted\nRMSE={rmse:.5f}, MAE={mae:.5f}, R²={r2:.3f}")
    axs[0].grid(True)
    axs[0].axis("equal")

    # === 2. Line plot ===
    axs[1].plot(trues, label="True", color="black", linewidth=1.5)
    axs[1].plot(preds, label="Predicted", color="blue", linestyle="--")
    axs[1].set_title("True vs Predicted (Index order)")
    axs[1].set_xlabel("Index")
    axs[1].set_ylabel("Ox(ppm)")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"✅ Kriging LOOCV plot saved to: {output_path}")

###############################################################################
def generate_kriging_image_with_labels_only(
    df,
    bounds,
    ox_min,
    ox_max,
    variogram_model='linear',
    transform=None,
    output_file='ox_kriging_labels.png',
    num_cells=800
):
    """
    Generate a Simple Kriging interpolated grayscale image with station locations and Ox labels (no map, no wind).

    Parameters:
        df: DataFrame with columns ['longitude', 'latitude', 'Ox(ppm)']
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        ox_min, ox_max: fixed grayscale scale
        variogram_model: variogram model type ('linear', 'spherical', etc.)
        output_file: path to save resulting PNG
        num_cells: resolution of interpolation grid
        transform: None, 'log', or 'sqrt' to apply to target variable
    """
    if ox_min is None or ox_max is None:
        raise ValueError("ox_min and ox_max must be provided to ensure consistent grayscale.")

    x = df['longitude'].values
    y = df['latitude'].values
    z_raw = df['Ox(ppm)'].values
    
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
    print(f"✅ Kriging label image saved to: {output_file}")

###############################################################################

# === CONFIG ===
data_dir = '..\\data\\Osaka\\'
prefecture_code = '27'
station_coordinates = 'Stations_Ox.csv'
csv_path = os.path.join(data_dir, station_coordinates)

# === Load station coordinates ===
df = pd.read_csv(csv_path, skipinitialspace=True)

# === Load and merge latest Ox data ===
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

models = ['linear', 'gaussian', 'exponential', 'spherical']
transforms = [None, 'log', 'sqrt']

print("Variogram  | Transform |   RMSE   |   MAE    |   R²")
print("-" * 50)
results = []

for model in models:
    for transform in transforms:
        rmse, mae, r2, trues, preds = evaluate_kriging_loocv(df, variogram_model=model, transform=transform)
        t_label = transform if transform else "none"
        print(f"{model:10s} | {t_label:9s} | {rmse:8.5f} | {mae:8.5f} | {r2:6.3f}")
        results.append((model, t_label, rmse, mae, r2))

best = max(results, key=lambda x: x[4])
best_model = best[0]
best_transform = best[1]
print(f"\n✅ Best model by R²: {best_model}, {best_transform} transform → RMSE={best[2]:.5f}, MAE={best[3]:.5f}, R²={best[4]:.3f}")

rmse, mae, r2, trues, preds = evaluate_kriging_loocv(df, variogram_model=best_model, transform=best_transform)
ox_kriging_loocv="ox_kriging_loocv.png"
plot_kriging_loocv_results(rmse, mae, r2, trues, preds, ox_kriging_loocv)

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

kriging_labels_image_path = 'ox_kriging_with_labels_only.png'
generate_kriging_image_with_labels_only(
    df,
    bounds,
    ox_min,
    ox_max,
    variogram_model=best_model,
    transform=best_transform,
    output_file=kriging_labels_image_path,
    num_cells=300
)

html_path = "OxMapSimpleKriging_LastHour.html"
screenshot_path = "map_screenshot.jpg"
capture_html_map_screenshot(html_path, screenshot_path)

save_kriging_report_pdf(
    output_path="kriging_report.pdf",
    results=results,  # lista di tuple (model, transform, rmse, mae, r2)
    formula_image_path=formula_image_path,
    html_screenshot_path=screenshot_path,
    additional_image_path=ox_kriging_loocv,
    kriging_labels_image_path=kriging_labels_image_path
)
