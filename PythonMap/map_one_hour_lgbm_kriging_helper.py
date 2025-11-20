# -*- coding: utf-8 -*-
"""
Created: 2025/11/16
Author: Mario
Description: Spatial LightGBM with LOOCV, map rendering, and PDF report.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from matplotlib.patches import Rectangle
from pykrige.ok import OrdinaryKriging
from scipy.spatial import cKDTree
from PIL import Image

from map_utils import idw_grid_vectorized

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

    n_estimators = 450
    max_depth = 12
    learning_rate = 0.05
    min_child_samples = 6
    colsample_bytree = 1.0
    subsample = 0.01
    num_leaves = 32

    preds, trues = [], []
    for i in range(len(df)):
        X_train = X_scaled_df.drop(i)
        y_train = np.delete(y, i)
        X_test = X_scaled_df.iloc[[i]]

        model = LGBMRegressor(
            objective='regression',
            boosting_type='gbdt',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            num_leaves=num_leaves,
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

    return model, residuals_df, scaler

###############################################################################
def kriging_interpolate_residuals(
    df,
    bounds,
    grid_resolution=200,
    variogram_model="linear",
    output_file="kriging_of_lgbm_residuals.png"
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
def generate_combined_confidence_overlay_image(
        df,
        model,
        scaler,
        features,
        bounds,
        target,
        vmin,
        vmax,
        output_file="combined_prediction.png",
        num_cells=300,
        max_distance_km=15,
        cmap_name="Reds",
        p=2  # IDW power
):
    """
    Combined LightGBM + IDW map with confidence mask and wind arrows.
    """

    # =========================================================
    # Grid setup
    # =========================================================
    (lat_min, lon_min), (lat_max, lon_max) = bounds

    grid_x = np.linspace(lon_min, lon_max, num_cells)
    grid_y = np.linspace(lat_min, lat_max, num_cells)
    grid_lon, grid_lat = np.meshgrid(grid_x, grid_y)

    flat_lon = grid_lon.ravel()
    flat_lat = grid_lat.ravel()

    # =========================================================
    # Build grid feature matrix (IDW for non-geo features)
    # =========================================================
    idw_maps = idw_grid_vectorized(df, grid_lon, grid_lat, features, p)
    
    feature_matrix = []
    for f in features:
        if f == "longitude":
            feature_matrix.append(grid_lon)
        elif f == "latitude":
            feature_matrix.append(grid_lat)
        else:
            feature_matrix.append(idw_maps[f])
    
    # Stack into shape (Ny * Nx, num_features)
    grid_df = pd.DataFrame(
        np.stack([m.ravel() for m in feature_matrix], axis=1),
        columns=features
    )

    # =========================================================
    # Scale features
    # =========================================================
    try:
        grid_scaled = pd.DataFrame(scaler.transform(grid_df), columns=features)
    except Exception as e:
        print(f"[ERROR] Scaling grid failed: {e}")
        return

    # =========================================================
    # LightGBM prediction
    # =========================================================
    lgbm_pred = model.predict(grid_scaled)

    # =========================================================
    # Combined prediction (final)
    # =========================================================
    combined = lgbm_pred.reshape(num_cells, num_cells)

    # =========================================================
    # Create RGBA image from colormap
    # =========================================================
    norm = np.clip((combined - vmin) / (vmax - vmin), 0, 1)
    cmap = plt.get_cmap(cmap_name)
    color_img = (cmap(norm)[..., :3] * 255).astype(np.uint8)

    # =========================================================
    # Confidence mask based on distance to nearest station
    # =========================================================
    coords_grid = np.column_stack([flat_lon, flat_lat])
    station_coords = np.column_stack([df["longitude"], df["latitude"]])

    tree = cKDTree(station_coords)
    dists, _ = tree.query(coords_grid, k=1)

    # convert degrees to km approx
    dist_km = (dists * 111).reshape((num_cells, num_cells))

    confidence = np.clip(1 - dist_km / max_distance_km, 0, 1)

    # Alpha mask parameters
    low, high = 0.45, 0.55
    gray_alpha = np.zeros_like(confidence)

    gray_alpha[confidence <= low] = 180
    gray_alpha[confidence >= high] = 0

    mask = (confidence > low) & (confidence < high)
    gray_alpha[mask] = ((high - confidence[mask]) / (high - low) * 180).astype(np.uint8)

    # grayscale overlay
    gray_overlay = np.zeros((num_cells, num_cells, 4), dtype=np.uint8)
    gray_overlay[..., :3] = 200
    gray_overlay[..., 3] = gray_alpha

    # final RGBA
    rgba = np.zeros((num_cells, num_cells, 4), dtype=np.uint8)
    rgba[..., :3] = color_img
    rgba[..., 3] = 255

    fg_alpha = gray_overlay[..., 3:4] / 255.0
    rgba[..., :3] = (
        gray_overlay[..., :3] * fg_alpha +
        rgba[..., :3] * (1 - fg_alpha)
    ).astype(np.uint8)

    # =========================================================
    # Save final image
    # =========================================================
    fig, ax = plt.subplots(figsize=(8, 8 * (lat_max - lat_min) / (lon_max - lon_min)), dpi=200)

    ax.imshow(rgba, extent=(lon_min, lon_max, lat_min, lat_max), origin="lower")

    ax.axis("off")

    # Bounding box
    rect = Rectangle(
        (df["longitude"].min(), df["latitude"].min()),
        df["longitude"].max() - df["longitude"].min(),
        df["latitude"].max() - df["latitude"].min(),
        linewidth=1.5, edgecolor="black", linestyle="--", facecolor="none"
    )
    ax.add_patch(rect)

    # Stations
    ax.scatter(
        df["longitude"], df["latitude"],
        c="black", edgecolor="white", s=60, label="Stations"
    )

    # Wind arrows
    if "U" in df.columns and "V" in df.columns:
        ax.quiver(
            df["longitude"], df["latitude"],
            df["U"], df["V"],
            angles="xy", scale_units="xy", scale=100.0,
            color="blue", width=0.003, label="Wind"
        )

    plt.savefig(output_file, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"✅ Combined prediction saved to {output_file}")

###############################################################################
def generate_lgbm_idw_labels_image(
    df,
    bounds,
    target,
    vmin,
    vmax,
    lgbm_model,
    scaler,
    features,
    output_file='labels_image.png',
    num_cells=800,
    p=2   # IDW power
):
    """
    Generate a grayscale image using LightGBM + IDW.
    The resulting map includes station labels with predicted values.

    Parameters:
        df: DataFrame with columns ['longitude','latitude',target,...]
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        target: name of target column (e.g. 'Ox')
        vmin, vmax: grayscale min/max
        lgbm_model: fitted LightGBM model
        scaler: fitted StandardScaler
        features: list of features used by LGBM
        output_file: filename
        num_cells: grid resolution
        p: IDW power
    """

    if vmin is None or vmax is None:
        raise ValueError("vmin and vmax must be provided.")

    # =========================================================
    # Create grid
    # =========================================================
    (lat_min, lon_min), (lat_max, lon_max) = bounds

    grid_x = np.linspace(lon_min, lon_max, num_cells)
    grid_y = np.linspace(lat_min, lat_max, num_cells)
    grid_lon, grid_lat = np.meshgrid(grid_x, grid_y)

    # =========================================================
    # Build grid feature matrix using IDW for non-geo features
    # =========================================================
    idw_maps = idw_grid_vectorized(df, grid_lon, grid_lat, features, p)
    
    feature_matrix = []
    for f in features:
        if f == "longitude":
            feature_matrix.append(grid_lon)
        elif f == "latitude":
            feature_matrix.append(grid_lat)
        else:
            feature_matrix.append(idw_maps[f])
    
    # Stack into shape (Ny * Nx, num_features)
    grid_df = pd.DataFrame(
        np.stack([m.ravel() for m in feature_matrix], axis=1),
        columns=features
    )

    # =========================================================
    # Scale features
    # =========================================================
    try:
        grid_scaled = pd.DataFrame(scaler.transform(grid_df), columns=features)
    except Exception as e:
        print(f"[ERROR] Failed to scale grid features: {e}")
        return

    # =========================================================
    # LightGBM predictions on grid
    # =========================================================
    lgbm_pred_grid = lgbm_model.predict(grid_scaled)
    combined_grid = lgbm_pred_grid.reshape(num_cells, num_cells)

    # =========================================================
    # Plot grayscale image
    # =========================================================
    aspect_ratio = (lat_max - lat_min) / (lon_max - lon_min)
    fig, ax = plt.subplots(figsize=(6, 6 * aspect_ratio), dpi=200)
    ax.axis("off")

    ax.imshow(
        combined_grid,
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin="lower",
        cmap="Greys",
        vmin=vmin,
        vmax=vmax,
        aspect="auto"
    )

    # =========================================================
    # Bounding box
    # =========================================================
    x = df["longitude"].values
    y = df["latitude"].values

    rect = Rectangle(
        (x.min(), y.min()),
        x.max() - x.min(),
        y.max() - y.min(),
        linewidth=1.5,
        edgecolor="black",
        linestyle="--",
        facecolor="none"
    )
    ax.add_patch(rect)

    # =========================================================
    # Station markers
    # =========================================================
    ax.scatter(x, y, c="black", edgecolor="white", s=50, zorder=5)

    # =========================================================
    # Value labels (LGBM)
    # =========================================================
    for i, row in df.iterrows():
        row_df = pd.DataFrame([row[features]])
        row_scaled = scaler.transform(row_df)
        pred_lgbm = lgbm_model.predict(row_scaled)[0]

        ax.text(
            row["longitude"] + 0.01,
            row["latitude"] + 0.005,
            f"{pred_lgbm:.3f}",
            fontsize=6,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
            zorder=6
        )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()

    print(f"✅ LGBM + IDW + Kriging (labels) image saved to: {output_file}")


###############################################################################
def save_lgbm_formula_as_jpg(filename="formula_lgbm_kriging.jpg"):
    """
    Save a visual explanation of the LightGBM + Kriging prediction as a JPEG image.

    Parameters:
        filename: output file path
    """
    formula = (
        r"$\hat{y}(x) = f_{\mathrm{LGBM}}(x) + r_{\mathrm{Kriging}}(x)$"
    )

    explanation_lines = [
        r"$\hat{y}(x)$: final predicted value at location $x$ (e.g., Ox concentration)",
        r"$f_{\mathrm{LGBM}}(x)$: prediction from the LightGBM model at $x$",
        r"$r_{\mathrm{Kriging}}(x)$: interpolated residual at $x$ using Ordinary Kriging",
        "",
        r"Step 1: Train LightGBM with LOOCV and compute residuals",
        r"Step 2: Fit Ordinary Kriging on residuals from training stations",
        r"Step 3: Predict on a spatial grid and combine the two terms",
        "",
        r"Kriging captures spatial patterns not learned by LightGBM."
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')

    # Title formula
    ax.text(0, 1, formula, fontsize=20, ha='left', va='center')

    # Explanation
    y_start = 0.75
    line_spacing = 0.07
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * line_spacing, line,
                fontsize=12, ha='left', va='center')

    plt.tight_layout()

    temp_file = "_temp_lgbm_kriging_formula.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

    # Convert to JPEG
    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved LGBM + Kriging formula JPEG to {filename}")
