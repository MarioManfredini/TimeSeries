# -*- coding: utf-8 -*-
"""
Created: 2025/11/16

Author: Mario
"""

import numpy as np

import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.patches import Rectangle

from map_utils import get_variogram_parameters

###############################################################################
def evaluate_kriging_loocv(target, df, variogram_model='linear', transform=None):
    """
    Evaluate LOOCV performance for Ordinary Kriging using a given variogram model,
    with optional transformation (e.g., 'log', 'sqrt').

    Parameters:
        df: DataFrame with 'longitude', 'latitude', measurements
        variogram_model: str, variogram model name
        transform: str or None. Options: 'log', 'sqrt', None

    Returns:
        rmse, mae, r2
    """
    epsilon = 1e-4  # For log transform

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
                coordinates_type="geographic",
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
def generate_kriging_image_with_labels_only(
    target,
    df,
    bounds,
    vmin,
    vmax,
    variogram_model='linear',
    transform=None,
    output_file='kriging_labels.png',
    num_cells=800
):
    """
    Generate a Simple Kriging interpolated grayscale image with
    station locations and measurement labels (no map, no wind).
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
