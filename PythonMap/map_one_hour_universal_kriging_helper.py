# -*- coding: utf-8 -*-
"""
Created: 2025/11/16

Author: Mario
"""

import numpy as np

import matplotlib.pyplot as plt
from pykrige.uk import UniversalKriging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata

from map_utils import (
    get_variogram_parameters
)

###############################################################################
def evaluate_universal_kriging_loocv(
    target,
    df,
    variogram_model="spherical",
    transform=None,
    drift_terms=None,
    x_col="longitude",
    y_col="latitude",
    variogram_parameters=None,
    debug=True
):
    if drift_terms is None or len(drift_terms) == 0:
        raise ValueError("drift_terms must be non-empty.")

    x_all = df[x_col].values
    y_all = df[y_col].values
    z_raw = df[target].values

    epsilon = 1e-4
    if transform == "log":
        z_all = np.log(z_raw + epsilon)
        inv = lambda x: np.exp(x) - epsilon
    elif transform == "sqrt":
        z_all = np.sqrt(z_raw)
        inv = lambda x: x * x
    else:
        z_all = z_raw
        inv = lambda x: x

    drift_all = df[drift_terms].values
    N, D = drift_all.shape

    if debug:
        print("====== GLOBAL DEBUG ======")
        print(f"Samples: {N}")
        print(f"Drift variables: {D}")
        print(f"drift_all shape: {drift_all.shape}")
        print(f"Any NaN in drift_all: {np.isnan(drift_all).any()}")
        print("==========================\n")

    preds, trues = [], []

    for i in range(N):

        if debug:
            print(f"\n===== LOOCV POINT {i} =====")

        x_train = np.delete(x_all, i)
        y_train = np.delete(y_all, i)
        z_train = np.delete(z_all, i)
        drift_train = np.delete(drift_all, i, axis=0)
        drift_test = drift_all[i, :]  # shape (D,)

        # --- specified_drift format ---
        specified_train = [drift_train[:, d] for d in range(D)]
        specified_test = [np.array([drift_test[d]]) for d in range(D)]

        if debug:
            print("Training drift arrays:", len(specified_train))
            print("Shape each:", specified_train[0].shape)
            print("Test drift arrays:", len(specified_test))
            print("Shape each:", specified_test[0].shape)

        try:
            uk = UniversalKriging(
                x_train,
                y_train,
                z_train,
                variogram_model=variogram_model,
                variogram_parameters=variogram_parameters,
                drift_terms=["specified"],
                specified_drift=specified_train,
                enable_plotting=False,
                verbose=False
            )

            z_pred_t, _ = uk.execute(
                "points",
                np.array([x_all[i]]),
                np.array([y_all[i]]),
                specified_drift_arrays=specified_test
            )

            z_pred = inv(z_pred_t[0])
            preds.append(z_pred)
            trues.append(z_raw[i])

            if debug:
                print(f"Pred transformed: {z_pred_t[0]:.6f}")
                print(f"Pred final     : {z_pred:.6f}")
                print(f"True value     : {z_raw[i]:.6f}")

        except Exception as e:
            print(f"❌ ERROR UK at point {i}: {e}")
            continue

    trues = np.array(trues)
    preds = np.array(preds)

    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)

    print("\n======= FINAL RESULTS =======")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"R²  : {r2:.6f}")
    print("==============================")

    return rmse, mae, r2, trues, preds


###############################################################################
def generate_universal_kriging_image_with_labels_only(
    target,
    df,
    bounds,
    vmin,
    vmax,
    drift_terms,
    variogram_model='gaussian',
    transform=None,
    output_file='uk_kriging_labels.png',
    num_cells=800
):
    """
    Generate a Universal Kriging interpolated grayscale image with station locations and labels,
    using external drift variables correctly through PyKrige's 'specified' drift mode.
    """
    # === Basic checks ===
    if vmin is None or vmax is None:
        raise ValueError("vmin and vmax must be provided.")

    if drift_terms is None or len(drift_terms) == 0:
        raise ValueError("drift_terms must be non-empty.")

    # Extract data
    x = df["longitude"].values
    y = df["latitude"].values
    z_raw = df[target].values
    drift_raw = df[drift_terms].values   # shape (N, D)

    N, D = drift_raw.shape

    # === Transform values ===
    epsilon = 1e-4
    if transform == "log":
        z = np.log(z_raw + epsilon)
        inv = lambda x: np.exp(x) - epsilon
    elif transform == "sqrt":
        z = np.sqrt(z_raw)
        inv = lambda x: x*x
    else:
        z = z_raw
        inv = lambda x: x

    # Each drift must be an array length N → list of D arrays
    specified_train = [drift_raw[:, d] for d in range(D)]

    # === Create interpolation grid ===
    (lat_min, lon_min), (lat_max, lon_max) = bounds
    grid_x = np.linspace(lon_min, lon_max, num_cells)
    grid_y = np.linspace(lat_min, lat_max, num_cells)
    GX, GY = np.meshgrid(grid_x, grid_y)

    # === Create drift grid arrays ===
    drift_grids = []
    for d in range(D):
        grid_d = griddata(
            points=(x, y),
            values=drift_raw[:, d],
            xi=(GX, GY),
            method='linear',
            fill_value=np.nan
        )
        drift_grids.append(grid_d)

    # === Fit Universal Kriging ===
    uk = UniversalKriging(
        x,
        y,
        z,
        variogram_model=variogram_model,
        variogram_parameters=None,
        drift_terms=["specified"],
        specified_drift=specified_train,   # TRAIN drift arrays
        verbose=False,
        enable_plotting=False
    )

    # === Kriging execution (with external drift grids) ===
    grid_z_transformed, _ = uk.execute(
        "grid",
        grid_x,
        grid_y,
        specified_drift_arrays=drift_grids  # TEST drift grids
    )
    grid_z = inv(grid_z_transformed)

    # === Plot ===
    aspect_ratio = (lat_max - lat_min) / (lon_max - lon_min)
    width = 6
    height = width * aspect_ratio
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    ax.axis("off")

    ax.imshow(
        grid_z,
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="Greys",
        alpha=0.9,
        aspect="auto"
    )

    # Bounding box
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

    # Stations and labels
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

    print(f"✅ Universal Kriging label image saved to: {output_file}")

###############################################################################
def generate_universal_kriging_grid(
    target,
    df,
    bounds,
    drift_terms,
    num_cells=300,
    variogram_model='linear'
):
    """
    Compute a 2D grid of interpolated values using Universal Kriging.

    Parameters:
        target: name of the column to interpolate (e.g., 'Ox(ppm)')
        df: DataFrame with 'longitude', 'latitude', measurements, and external drift variables
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        drift_terms: list of column names to use as external drift (e.g., ['NO(ppm)', 'NO2(ppm)'])
        num_cells: number of grid cells along the longitude axis
        variogram_model: variogram model name (e.g. 'linear', 'power', 'gaussian')

    Returns:
        grid: interpolated grid (lat_cells x lon_cells)
    """

    print("\n=== [UK GRID] Starting Universal Kriging grid generation ===")

    # === Basic checks ===
    if drift_terms is None or len(drift_terms) == 0:
        raise ValueError("drift_terms must be a non-empty list.")

    print(f"[UK GRID] Target variable: {target}")
    print(f"[UK GRID] Drift terms: {drift_terms}")
    print(f"[UK GRID] Variogram model: {variogram_model}")
    print(f"[UK GRID] Requested grid resolution (longitude cells): {num_cells}")

    # === Extract coordinates and values ===
    print("[UK GRID] Extracting coordinates and measured values...")
    x = df['longitude'].values
    y = df['latitude'].values
    z = df[target].values
    external_drift = df[drift_terms].values

    print(f"[UK GRID] Number of stations: {len(df)}")

    # === Unpack bounds ===
    (lat_min, lon_min), (lat_max, lon_max) = bounds
    print(f"[UK GRID] Bounds: lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]")

    # === Compute grid dimensions to preserve square cells ===
    lon_cells = num_cells
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    lat_cells = int(lon_cells * lat_range / lon_range)

    print(f"[UK GRID] Grid dimensions → lat_cells={lat_cells}, lon_cells={lon_cells}")

    # === Generate grid arrays ===
    print("[UK GRID] Generating interpolation meshgrid...")
    grid_lon = np.linspace(lon_min, lon_max, lon_cells)
    grid_lat = np.linspace(lat_min, lat_max, lat_cells)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

    # === Get variogram parameters ===
    print("[UK GRID] Getting variogram parameters...")
    variogram_parameters = get_variogram_parameters(variogram_model)
    print(f"[UK GRID] Variogram parameters: {variogram_parameters}")

    # === Initialize Universal Kriging ===
    print("[UK GRID] Initializing UniversalKriging model...")
    uk = UniversalKriging(
        x,
        y,
        z,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        drift_terms=drift_terms,
        external_drift=external_drift,
        verbose=False,
        enable_plotting=False
    )

    # === Execute Kriging ===
    print("[UK GRID] Executing kriging interpolation...")
    z_interp, _ = uk.execute(
        style='grid',
        xpoints=grid_lon,
        ypoints=grid_lat
    )

    print("[UK GRID] Interpolation completed.")

    # === Convert to ndarray ===
    grid = np.array(z_interp)
    print(f"[UK GRID] Generated grid shape: {grid.shape}")

    print("=== [UK GRID] Universal Kriging grid generation completed ===\n")

    return grid

