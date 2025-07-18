# -*- coding: utf-8 -*-
"""
Created: 2025/07/12
Author: Mario
Description: Spatial LightGBM with LOOCV, map rendering, and PDF report.
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
from map_report import capture_html_map_screenshot, save_map_report_pdf, save_lgbm_kriging_formula_as_jpg
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
from pykrige.ok import OrdinaryKriging
from scipy.spatial import cKDTree

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
    axs[0].set_title(f"LightGBM LOOCV - True vs Predicted\nRMSE={rmse:.5f}, MAE={mae:.5f}, R²={r2:.3f}")
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
    print(f"✅ LightGBM LOOCV plot saved to: {output_path}")

###############################################################################
def compute_lgbm_loocv_residuals(df, features, target='Ox(ppm)'):
    """
    Perform LOOCV with LightGBM and return a DataFrame containing residuals.

    Parameters:
        df: pandas DataFrame with station data
        features: list of features to use for prediction
        target: target variable name (default 'Ox(ppm)')

    Returns:
        DataFrame with columns:
        ['station_code', 'latitude', 'longitude', target, 'prediction', 'residual']
    """
    X = df[features]
    y = df[target].values

    scaler = StandardScaler()
    X_scaled_df = pd.DataFrame(scaler.fit_transform(X), columns=features)

    preds, trues = [], []
    for i in range(len(df)):
        X_train = X_scaled_df.drop(i)
        y_train = np.delete(y, i)
        X_test = X_scaled_df.iloc[[i]]

        model = LGBMRegressor(
            objective='regression',
            boosting_type='gbdt',
            n_estimators=400,
            learning_rate=0.02,
            max_depth=-1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        preds.append(y_pred[0])
        trues.append(y[i])

    residuals_df = df[['station_code', 'latitude', 'longitude', target]].copy()
    residuals_df['prediction'] = preds
    residuals_df['residual'] = residuals_df[target] - residuals_df['prediction']

    return model, residuals_df

###############################################################################
def kriging_interpolate_residuals(
    df,
    bounds,
    grid_resolution=200,
    variogram_model="linear",
    output_file="ox_kriging_of_lgbm_residuals.png"
):
    """
    Performs Ordinary Kriging interpolation on residuals.

    Parameters:
        df: DataFrame with columns ['longitude', 'latitude', 'residual']
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        grid_resolution: number of cells along each axis
        variogram_model: 'linear', 'power', 'gaussian', 'spherical', or 'exponential'
        output_file: Path to save resulting image
    """
    # Extract coordinates and residuals
    lons = df["longitude"].values
    lats = df["latitude"].values
    residuals = df["residual"].values

    # Grid coordinates
    (lat_min, lon_min), (lat_max, lon_max) = bounds
    grid_lon = np.linspace(lon_min, lon_max, grid_resolution)
    grid_lat = np.linspace(lat_min, lat_max, grid_resolution)

    # Kriging interpolation
    OK = OrdinaryKriging(
        lons,
        lats,
        residuals,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
        coordinates_type="geographic"
    )

    z, ss = OK.execute("grid", grid_lon, grid_lat)

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    c = ax.imshow(
        z,
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin="lower",
        cmap="coolwarm",
        alpha=0.85
    )
    plt.colorbar(c, ax=ax, label="Residuals")
    ax.set_title("Kriging Interpolation of LGBM Residuals")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.scatter(lons, lats, c="black", s=40, edgecolors="white", label="Stations")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"✅ Kriging residual map saved to: {output_file}")

###############################################################################
def generate_ox_combined_confidence_overlay_image(
    df,
    model,
    scaler,
    features,
    residuals_df,
    bounds,
    ox_min,
    ox_max,
    output_file="ox_lgbm_kriging_overlay.png",
    num_cells=300,
    max_distance_km=15,
    cmap_name="Greys",
    variogram_model="linear"
):
    """
    Generates a combined LightGBM + Kriging residuals prediction overlay image for folium.

    Parameters:
        df: DataFrame with stations data
        model: trained LGBMRegressor
        scaler: fitted scaler
        features: list of model features
        residuals_df: DataFrame with 'longitude', 'latitude', 'residual'
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        ox_min, ox_max: color scale bounds
        output_file: path to save the image
        num_cells: number of grid cells
        max_distance_km: max distance to define confidence fade
        cmap_name: matplotlib colormap
        variogram_model: type of kriging variogram
    """
    x = df['longitude'].values
    y = df['latitude'].values
    (lat_min, lon_min), (lat_max, lon_max) = bounds

    # === Grid definition ===
    grid_lon = np.linspace(lon_min, lon_max, num_cells)
    grid_lat = np.linspace(lat_min, lat_max, num_cells)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
    flat_lon = grid_lon_mesh.ravel()
    flat_lat = grid_lat_mesh.ravel()

    # === LightGBM prediction on grid ===
    mean_values = {f: df[f].mean() for f in features if f not in ['longitude', 'latitude'] and f in df.columns}
    feature_grid = []
    for lon, lat in zip(flat_lon, flat_lat):
        row = []
        for f in features:
            if f == 'longitude':
                row.append(lon)
            elif f == 'latitude':
                row.append(lat)
            else:
                row.append(mean_values.get(f, 0))
        feature_grid.append(row)

    X_grid = pd.DataFrame(feature_grid, columns=features)
    X_scaled = scaler.transform(X_grid)
    lgbm_preds = model.predict(X_scaled).reshape((num_cells, num_cells))

    # === Kriging of residuals ===
    model = OrdinaryKriging(
        residuals_df["longitude"],
        residuals_df["latitude"],
        residuals_df["residual"],
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
        coordinates_type="geographic"
    )
    kriged_residuals, _ = model.execute("grid", grid_lon, grid_lat)

    # === Combined prediction ===
    combined = lgbm_preds + kriged_residuals

    # === Generate RGBA image with confidence
    norm = np.clip((combined - ox_min) / (ox_max - ox_min), 0, 1)
    cmap = plt.get_cmap(cmap_name)
    color_img = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)

    # === Confidence based on distance from stations
    coords_grid = np.column_stack([flat_lon, flat_lat])
    station_coords = np.column_stack([df["longitude"], df["latitude"]])
    tree = cKDTree(station_coords)
    dists, _ = tree.query(coords_grid, k=1)
    dist_km = dists * 111  # rough conversion
    confidence = np.clip(1 - dist_km / max_distance_km, 0, 1).reshape((num_cells, num_cells))

    # === Alpha blending (grayscale overlay for low confidence)
    low, high = 0.45, 0.55
    gray_alpha = np.zeros_like(confidence)
    gray_alpha[confidence <= low] = 180
    gray_alpha[confidence >= high] = 0
    mask = (confidence > low) & (confidence < high)
    gray_alpha[mask] = ((high - confidence[mask]) / (high - low) * 180).astype(np.uint8)

    gray_overlay = np.zeros((num_cells, num_cells, 4), dtype=np.uint8)
    gray_overlay[..., :3] = 200
    gray_overlay[..., 3] = gray_alpha

    rgba = np.zeros((num_cells, num_cells, 4), dtype=np.uint8)
    rgba[..., :3] = color_img
    rgba[..., 3] = 255

    fg_alpha = gray_overlay[..., 3:4] / 255.0
    rgba[..., :3] = (
        gray_overlay[..., :3] * fg_alpha + rgba[..., :3] * (1 - fg_alpha)
    ).astype(np.uint8)

    # === Save image ===
    fig, ax = plt.subplots(figsize=(8, 8 * (lat_max - lat_min) / (lon_max - lon_min)), dpi=200)
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
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"✅ Combined overlay image saved to: {output_file}")
    
    return model

###############################################################################
def generate_lgbm_kriging_labels_image(
    df,
    bounds,
    ox_min,
    ox_max,
    lgbm_model,
    kriging_model,
    scaler,
    features,
    output_file='ox_lgbm_kriging_labels.png',
    num_cells=800
):
    """
    Generate a grayscale image with predicted Ox values using LightGBM + Kriging residuals.

    Parameters:
        df: DataFrame with required features and columns ['longitude', 'latitude', 'Ox(ppm)']
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        ox_min, ox_max: grayscale min/max range
        lgbm_model: trained LGBMRegressor
        kriging_model: trained OrdinaryKriging on residuals
        scaler: fitted scaler (StandardScaler or similar)
        features: list of feature names used in LGBM training
        output_file: file path to save image
        num_cells: resolution of the grid
    """
    if ox_min is None or ox_max is None:
        raise ValueError("ox_min and ox_max must be provided.")

    (lat_min, lon_min), (lat_max, lon_max) = bounds

    # === Create grid ===
    grid_x = np.linspace(lon_min, lon_max, num_cells)
    grid_y = np.linspace(lat_min, lat_max, num_cells)
    grid_lon, grid_lat = np.meshgrid(grid_x, grid_y)

    flat_lon = grid_lon.ravel()
    flat_lat = grid_lat.ravel()

    # === Compute mean feature values ===
    mean_values = {f: df[f].mean() for f in features if f in df.columns}
    missing = set(features) - set(mean_values)
    if missing:
        print(f"[WARN] Some features not found in df, using 0: {missing}")

    # === Prepare feature vectors for prediction ===
    feature_vectors = []
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
                row.append(0)
        feature_vectors.append(row)

    grid_df = pd.DataFrame(feature_vectors, columns=features)

    # === Apply scaler
    try:
        grid_scaled = pd.DataFrame(scaler.transform(grid_df), columns=features)
    except Exception as e:
        print(f"[ERROR] Failed to scale grid features: {e}")
        return

    # === Predict with LGBM
    y_lgbm_pred = lgbm_model.predict(grid_scaled)

    # === Predict residuals with Kriging
    try:
        y_kriging_pred, _ = kriging_model.execute(
            "grid", grid_lon[0], grid_lat[:, 0]
        )
        y_kriging_pred = y_kriging_pred.data.reshape(-1)
    except Exception as e:
        print(f"[ERROR] Kriging prediction failed: {e}")
        y_kriging_pred = np.zeros_like(y_lgbm_pred)

    # === Combine predictions
    y_combined = y_lgbm_pred + y_kriging_pred
    grid_z = y_combined.reshape((num_cells, num_cells))

    # === Plot
    aspect_ratio = (lat_max - lat_min) / (lon_max - lon_min)
    fig, ax = plt.subplots(figsize=(6, 6 * aspect_ratio), dpi=200)
    ax.axis('off')

    ax.imshow(
        grid_z,
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin='lower',
        cmap='Greys',
        vmin=ox_min,
        vmax=ox_max,
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

    # Scatter station points
    ax.scatter(x, y, c='black', edgecolor='white', s=50, zorder=5)

    # Labels with combined predictions
    for i, row in df.iterrows():
        X_row = pd.DataFrame([row[features]])
        X_scaled = scaler.transform(X_row)
        pred_lgbm = lgbm_model.predict(X_scaled)[0]
        pred_kriging, _ = kriging_model.execute("points", [row['longitude']], [row['latitude']])
        final_estimate = pred_lgbm + pred_kriging[0]

        ax.text(
            row['longitude'] + 0.01,
            row['latitude'] + 0.005,
            f"{final_estimate:.3f}",
            fontsize=6,
            color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1),
            zorder=6
        )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    print(f"✅ Combined prediction label image saved to: {output_file}")

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

X = df[features]
y = df[target].values

# === Scaler initialization ===
scaler = StandardScaler()
X_scaled_df = pd.DataFrame(scaler.fit_transform(X), columns=features)

lgbm_model, residuals_df = compute_lgbm_loocv_residuals(df, features, target)

# Calcolo delle metriche
mse = mean_squared_error(residuals_df[target], residuals_df['prediction'])
rmse = mse ** 0.5
mae = mean_absolute_error(residuals_df[target], residuals_df['prediction'])
r2 = r2_score(residuals_df[target], residuals_df['prediction'])

print("\n✅ LightGBM LOOCV:")
print(f"RMSE: {rmse:.5f}, MAE: {mae:.5f}, R²: {r2:.3f}")

# === LOOCV ===
ox_lgbm_loocv="loocv.png"
plot_rf_loocv_results(rmse, mae, r2, residuals_df[target], residuals_df['prediction'], ox_lgbm_loocv)

# === Generate image ===
bounds = compute_bounds_from_df(df, margin_ratio=0.10)

kriging_interpolate_residuals(residuals_df, bounds)

ox_min = np.percentile(df[target], 5)
ox_max = np.percentile(df[target], 95)

combined_overlay_img = "confidence_overlay_image.png"
kriging_model = generate_ox_combined_confidence_overlay_image(
    df,
    lgbm_model,
    scaler,
    features,
    residuals_df,
    bounds,
    ox_min,
    ox_max,
    output_file=combined_overlay_img,
    num_cells=300,
    max_distance_km=15,
    cmap_name="Reds",
    variogram_model="linear"
)

center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
m_combined = folium.Map(location=[center_lat, center_lon], zoom_start=10)

ImageOverlay(
    name="LGBM + Kriging Residuals",
    image=combined_overlay_img,
    bounds=[[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]],
    opacity=0.7
).add_to(m_combined)

html_file = "OxMapLGBM_Kriging_OneHour.html"
m_combined.save(html_file)
print(f"✅ Folium map saved to: {html_file}")

formula_image_path = 'formula.jpg'
save_lgbm_kriging_formula_as_jpg(formula_image_path)

labels_image_path = 'labels_image.png'
generate_lgbm_kriging_labels_image(
    df=df,
    bounds=bounds,
    ox_min=ox_min,
    ox_max=ox_max,
    lgbm_model=lgbm_model,
    kriging_model=kriging_model,
    scaler=scaler,
    features=features,
    output_file=labels_image_path,
    num_cells=300
)

screenshot_path = "screenshot.jpg"
capture_html_map_screenshot(html_file, screenshot_path)

# === Report PDF ===
save_map_report_pdf(
    output_path="LightGBM_Kriging_Report.pdf",
    results=[(rmse, mae, r2, f"All stations ({year}/{month}/{day} {hour}H)")],
    formula_image_path=formula_image_path,
    html_screenshot_path=screenshot_path,
    labels_image_path=labels_image_path,
    additional_image_path=ox_lgbm_loocv,
    title=f"LightGBM Interpolation + residuals Kriging - {year}/{month}/{day} {hour:02d}H"
)
