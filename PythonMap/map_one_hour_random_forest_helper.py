# -*- coding: utf-8 -*-
"""
Created: 2025/11/16
Author: Mario
Description: Spatial Random Forest with LOOCV, map rendering, and PDF report.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from PIL import Image

###############################################################################
def generate_rf_image_with_labels_only(
    target,
    df,
    bounds,
    vmin,
    vmax,
    model,
    scaler,
    features,
    output_file='labels_image.png',
    num_cells=800
):
    """
    Generate a grayscale image with predicted values using Random Forest.

    Parameters:
        df: DataFrame with required features and columns ['longitude', 'latitude', measurements]
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        vmin, vmax: fixed grayscale range for consistency
        model: trained RandomForestRegressor
        scaler: fitted scaler (StandardScaler or MinMaxScaler)
        features: list of features used during training
        output_file: path to save PNG
        num_cells: resolution of grid
    """
    if vmin is None or vmax is None:
        raise ValueError("vmin and vmax must be provided.")

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
        vmin=vmin,
        vmax=vmax,
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
        value = row.get(target, None)
        if pd.notna(value):
            ax.text(
                row['longitude'] + 0.01,
                row['latitude'] + 0.005,
                f"{value:.3f}",
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
def generate_rf_confidence_overlay_image(
    df,
    bounds,
    model,
    features,
    vmin,
    vmax,
    output_file="prediction.png",
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
        vmin, vmax: min and max values for color scaling
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
    norm = np.clip((grid_z - vmin) / (vmax - vmin), 0, 1)
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
def save_rf_formula_as_jpg(filename="formula_rf.jpg"):
    """
    Save a visual explanation of the Random Forest model as a JPEG image.

    Parameters:
        filename: output file path
    """
    formula = (
        r"$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} f_t(x)$"
    )

    explanation_lines = [
        r"$\hat{y}$: predicted value (e.g., Ox concentration)",
        r"$T$: total number of trees in the forest",
        r"$f_t(x)$: prediction of tree $t$ for input $x$",
        r"$x$: input features (e.g., NO, NO₂, U, V, longitude, latitude)",
        "",
        r"Each tree is trained on a bootstrap sample",
        r"and uses a random subset of features at each split.",
        r"Final prediction is the average of all tree outputs."
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')

    # Title formula
    ax.text(0, 1, formula, fontsize=18, ha='left', va='center')

    # Explanation
    y_start = 0.75
    line_spacing = 0.07
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * line_spacing, line,
                fontsize=12, ha='left', va='center')

    plt.tight_layout()

    temp_file = "_temp_rf_formula.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

    # Convert to JPEG
    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved RF formula JPEG to {filename}")
