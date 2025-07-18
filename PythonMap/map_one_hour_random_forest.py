# -*- coding: utf-8 -*-
"""
Created: 2025/06/29
Author: Mario
Description: Spatial Random Forest with LOOCV, map rendering, and PDF report.
"""
import os
import pandas as pd
import numpy as np
import folium
from folium.raster_layers import ImageOverlay
from map_utils import (
    load_station_temporal_features,
    compute_bounds_from_df
)
from map_report import capture_html_map_screenshot, save_map_report_pdf, save_rf_formula_as_jpg
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler

###############################################################################
def plot_rf_loocv_results(rmse, mae, r2, trues, preds, output_path="ox_rf_loocv.jpg"):
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
    axs[0].set_title(f"Random Forest LOOCV - True vs Predicted\nRMSE={rmse:.5f}, MAE={mae:.5f}, R²={r2:.3f}")
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
    print(f"✅ Random Forest LOOCV plot saved to: {output_path}")

###############################################################################
def generate_rf_image_with_labels_only(
    df,
    bounds,
    ox_min,
    ox_max,
    model,
    scaler,
    features,
    output_file='ox_rf_labels.png',
    num_cells=800
):
    """
    Generate a grayscale image with predicted Ox values using Random Forest.

    Parameters:
        df: DataFrame with required features and columns ['longitude', 'latitude', 'Ox(ppm)']
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        ox_min, ox_max: fixed grayscale range for consistency
        model: trained RandomForestRegressor
        scaler: fitted scaler (StandardScaler or MinMaxScaler)
        features: list of features used during training
        output_file: path to save PNG
        num_cells: resolution of grid
    """
    if ox_min is None or ox_max is None:
        raise ValueError("ox_min and ox_max must be provided.")

    (lat_min, lon_min), (lat_max, lon_max) = bounds

    # === Create grid ===
    grid_x = np.linspace(lon_min, lon_max, num_cells)
    grid_y = np.linspace(lat_min, lat_max, num_cells)
    grid_lon, grid_lat = np.meshgrid(grid_x, grid_y)

    # Flatten grid
    flat_lon = grid_lon.ravel()
    flat_lat = grid_lat.ravel()

    # === Compute mean feature values ===
    mean_values = {}
    for f in features:
        if f in df.columns:
            mean_values[f] = df[f].mean()
    missing = set(features) - set(mean_values)
    if missing:
        print(f"[WARN] Some features not found in df, skipped: {missing}")

    # === Build grid feature vectors ===
    features_grid = []
    for lon, lat in zip(flat_lon, flat_lat):
        row = []
        for f in features:
            if f == 'longitude':
                row.append(lon)
            elif f == 'latitude':
                row.append(lat)
            elif f in mean_values:
                row.append(mean_values[f])
            else:
                row.append(0)  # default if feature not available
        features_grid.append(row)

    features_grid = np.array(features_grid)

    # === Apply scaling
    try:
        features_scaled = scaler.transform(features_grid)
    except Exception as e:
        print(f"[ERROR] Failed to apply scaler to feature grid: {e}")
        return

    predictions = model.predict(features_scaled)
    grid_z = predictions.reshape((num_cells, num_cells))

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

    # Bounding box
    x = df['longitude'].values
    y = df['latitude'].values
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

    # Markers and labels
    ax.scatter(x, y, c='black', edgecolor='white', s=50, zorder=5)
    for i, row in df.iterrows():
        ox_value = row.get('Ox(ppm)', None)
        if pd.notna(ox_value):
            ax.text(
                row['longitude'] + 0.01,
                row['latitude'] + 0.005,
                f"{ox_value:.3f}",
                fontsize=6,
                color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1),
                zorder=6
            )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_file, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ RF label image saved to: {output_file}")


###############################################################################
def generate_ox_rf_confidence_overlay_image(
    df,
    bounds,
    model,
    features,
    ox_min,
    ox_max,
    output_file="ox_rf_prediction.png",
    num_cells=300,
    max_distance_km=20,
    cmap_name="Reds",
    scaler=None
):
    """
    Generates and saves a spatial prediction image using a trained model.

    Parameters:
        df: pandas DataFrame containing the station data and required features
        bounds: ((lat_min, lon_min), (lat_max, lon_max)) bounds for the map
        model: trained model
        features: list of feature names to use in prediction
        ox_min, ox_max: min and max values for color scaling
        output_file: path to save the output PNG image
        num_cells: number of grid cells along the x-axis
        max_distance_km: max distance considered reliable
        cmap_name: matplotlib colormap name
        scaler: optional fitted scaler to transform the features
    """
    x = df['longitude'].values
    y = df['latitude'].values
    (lat_min, lon_min), (lat_max, lon_max) = bounds

    lon_cells = num_cells
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    lat_cells = int(lon_cells * lat_range / lon_range)

    grid_x, grid_y = np.meshgrid(
        np.linspace(lon_min, lon_max, lon_cells),
        np.linspace(lat_min, lat_max, lat_cells)
    )
    grid_coords = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    grid_df = pd.DataFrame(grid_coords, columns=['longitude', 'latitude'])

    # === Fill default values for expected predictors
    default_features = {
        'U': df['U'].mean() if 'U' in df.columns else 0,
        'V': df['V'].mean() if 'V' in df.columns else 0,
        'NO(ppm)': df['NO(ppm)'].mean() if 'NO(ppm)' in df.columns else 0,
        'NO2(ppm)': df['NO2(ppm)'].mean() if 'NO2(ppm)' in df.columns else 0,
        'hour_sin': df['hour_sin'].mean() if 'hour_sin' in df.columns else 0,
        'hour_cos': df['hour_cos'].mean() if 'hour_cos' in df.columns else 0,
        'dayofweek': df['dayofweek'].mode()[0] if 'dayofweek' in df.columns else 0,
        'is_weekend': df['is_weekend'].mode()[0] if 'is_weekend' in df.columns else 0,
    }

    extra_df = pd.DataFrame(
        {col: value for col, value in default_features.items()},
        index=grid_df.index
    )
    grid_df = pd.concat([grid_df, extra_df], axis=1)

    # === Usa solo le feature che esistono davvero nella griglia
    valid_features = [f for f in features if f in grid_df.columns]
    missing = set(features) - set(valid_features)
    if missing:
        print(f"[WARN] Missing features in grid, skipped: {missing}")

    missing_features = [f for f in features if f not in grid_df.columns]
    if missing_features:
        missing_df = pd.DataFrame(0, index=grid_df.index, columns=missing_features)
        grid_df = pd.concat([grid_df, missing_df], axis=1)
    
    # Ordina le colonne nello stesso ordine del training
    X_pred_df = grid_df[features].copy()
    
    # === Applica scaler
    if scaler is not None:
        try:
            X_pred = pd.DataFrame(scaler.transform(X_pred_df), columns=X_pred_df.columns)
        except Exception as e:
            print(f"[ERROR] Failed to apply scaler: {e}")
            return
    else:
        X_pred = X_pred_df.values

    y_pred = model.predict(X_pred)
    grid_z = y_pred.reshape((lat_cells, lon_cells))

    # === Colori
    norm = np.clip((grid_z - ox_min) / (ox_max - ox_min), 0, 1)
    cmap = plt.get_cmap(cmap_name)
    color_img = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)

    # === Distanza per confidence
    from scipy.spatial import cKDTree

    tree = cKDTree(np.vstack((x, y)).T)
    dists, _ = tree.query(grid_coords, k=1)
    dist_km = dists * 111  # approx conversion
    confidence = np.clip(1 - dist_km / max_distance_km, 0, 1).reshape((lat_cells, lon_cells))

    # === Overlay grigia
    low, high = 0.45, 0.55
    gray_alpha = np.zeros_like(confidence)
    gray_alpha[confidence <= low] = 180
    gray_alpha[confidence >= high] = 0
    fade = (high - confidence[(confidence > low) & (confidence < high)]) / (high - low)
    gray_alpha[(confidence > low) & (confidence < high)] = (fade * 180).astype(np.uint8)

    gray_overlay = np.zeros((lat_cells, lon_cells, 4), dtype=np.uint8)
    gray_overlay[..., :3] = 200
    gray_overlay[..., 3] = gray_alpha

    rgba = np.zeros((lat_cells, lon_cells, 4), dtype=np.uint8)
    rgba[..., :3] = color_img
    rgba[..., 3] = 255

    fg_alpha = gray_overlay[..., 3:4] / 255.0
    rgba[..., :3] = (
        gray_overlay[..., :3] * fg_alpha + rgba[..., :3] * (1 - fg_alpha)
    ).astype(np.uint8)

    # === Plot
    fig, ax = plt.subplots(figsize=(8, 8 * lat_range / lon_range), dpi=200)
    ax.imshow(rgba, extent=(lon_min, lon_max, lat_min, lat_max), origin='lower')
    ax.axis('off')

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
    ax.scatter(x, y, c='black', edgecolor='white', s=60, label='Stations')

    if 'U' in df.columns and 'V' in df.columns:
        ax.quiver(
            df['longitude'], df['latitude'],
            df['U'], df['V'],
            angles='xy', scale_units='xy', scale=100.0,
            color='blue', width=0.003, label='Wind'
        )

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Random Forest prediction image saved to: {output_file}")

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

# Exclude non-numeric columns like 'datetime' and 'WD(16Dir)'
exclude_cols = ['datetime', 'WD(16Dir)', 'station_code', 'station_name'] 
features = [col for col in df.columns if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)]

X = df[features].values
y = df[target].values

# === Scaler initialization ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

preds, trues = [], []
for i in range(len(df)):
    X_train = np.delete(X_scaled, i, axis=0)
    y_train = np.delete(y, i)
    X_test = X_scaled[i].reshape(1, -1)

    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    preds.append(y_pred[0])
    trues.append(y[i])

mse = mean_squared_error(trues, preds)
rmse = mse ** 0.5
mae = mean_absolute_error(trues, preds)
r2 = r2_score(trues, preds)
print("\n✅ Random Forest LOOCV:")
print(f"RMSE: {rmse:.5f}, MAE: {mae:.5f}, R²: {r2:.3f}")

# === LOOCV ===
ox_rf_loocv="loocv.png"
plot_rf_loocv_results(rmse, mae, r2, trues, preds, ox_rf_loocv)

# === Generate image ===
bounds = compute_bounds_from_df(df, margin_ratio=0.10)
ox_min = np.percentile(df[target], 5)
ox_max = np.percentile(df[target], 95)
rf_image_path = "prediction.png"
generate_ox_rf_confidence_overlay_image(
    df,
    bounds,
    model,
    features,
    ox_min,
    ox_max,
    output_file=rf_image_path,
    num_cells=300,
    max_distance_km=15,
    cmap_name="Reds",
    scaler=scaler
)

# === Folium Map ===
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
image_overlay = ImageOverlay(
    name=f"Random Forest Predicted {target}",
    image=rf_image_path,
    bounds=[[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]],
    opacity=0.7
)
image_overlay.add_to(m)

html_file = "OxMapRF_LastHour.html"
m.save(html_file)
print(f"Map saved to: {html_file}")

formula_image_path = 'formula.jpg'
save_rf_formula_as_jpg(formula_image_path)

labels_image_path = 'labels_image.png'
generate_rf_image_with_labels_only(
    df,
    bounds,
    ox_min,
    ox_max,
    model,
    scaler,
    features,
    output_file=labels_image_path,
    num_cells=300
)

screenshot_path = "screenshot.jpg"
capture_html_map_screenshot(html_file, screenshot_path)

# === Report PDF ===
save_map_report_pdf(
    output_path="random_forest_report.pdf",
    results=[(rmse, mae, r2, f"All stations ({year}/{month}/{day} {hour}H)")],
    formula_image_path=formula_image_path,
    html_screenshot_path=screenshot_path,
    labels_image_path=labels_image_path,
    additional_image_path=ox_rf_loocv,
    title=f"Random Forest Spatial Interpolation - {year}/{month}/{day} {hour:02d}H"
)
