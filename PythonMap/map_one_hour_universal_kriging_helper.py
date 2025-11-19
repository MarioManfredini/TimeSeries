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
    Generate a Universal Kriging interpolated grayscale image with station locations and labels
    (no basemap, no wind visualization).

    Parameters:
        df: DataFrame with ['longitude', 'latitude', target, external drift variables]
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        vmin, vmax: fixed grayscale range for consistent intensity
        drift_terms: list of column names used as external drift
        variogram_model: variogram model ('gaussian', 'spherical', etc.)
        transform: None, 'log', or 'sqrt'
        output_file: PNG file path
        num_cells: grid resolution
    """

    # === Basic checks ===
    if vmin is None or vmax is None:
        raise ValueError("vmin and vmax must be provided for consistent grayscale rendering.")

    if drift_terms is None or len(drift_terms) == 0:
        raise ValueError("drift_terms must be a non-empty list of external drift variables.")

    # Extract data
    x = df['longitude'].values
    y = df['latitude'].values
    z_raw = df[target].values
    external_drift = df[drift_terms].values

    # === Apply transformation ===
    epsilon = 1e-4
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

    # === Create interpolation grid ===
    grid_x = np.linspace(lon_min, lon_max, num_cells)
    grid_y = np.linspace(lat_min, lat_max, num_cells)

    # Create drift grid
    from scipy.interpolate import griddata
    drift_grids = []
    for d in range(external_drift.shape[1]):
        drift_grids.append(
            griddata(
                points=(x, y),
                values=external_drift[:, d],
                xi=np.meshgrid(grid_x, grid_y),
                method='linear',
                fill_value=np.nan
            )
        )

    drift_grids = np.array(drift_grids)

    # === Fit Universal Kriging ===
    uk = UniversalKriging(
        x, y, z,
        variogram_model=variogram_model,
        variogram_parameters=None,
        drift_terms=drift_terms,
        external_drift=external_drift,
        verbose=False,
        enable_plotting=False
    )

    # === Execute kriging ===
    grid_z_transf, _ = uk.execute(
        'grid',
        grid_x,
        grid_y
    )

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

    # Bounding box of measured region
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

    print(f"✅ Universal Kriging label image saved to: {output_file}")
