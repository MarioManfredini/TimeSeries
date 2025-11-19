# -*- coding: utf-8 -*-
"""
Created: 2025/11/16

Author: Mario
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial import cKDTree
from matplotlib.patches import Rectangle
from PIL import Image

###############################################################################
def generate_idw_image_with_labels_only(
        target,
        df,
        bounds,
        grid,
        vmin,
        vmax,
        k=7,
        power=1,
        output_file='labels_image.png',
        num_cells=300
):
    """
    Generate an IDW interpolated grayscale image with station locations and measurements labels (no map, no wind).
    """
    if vmin is None or vmax is None:
        raise ValueError("vmin and vmax must be provided to ensure consistent grayscale.")

    # === Plot ===
    (lat_min, lon_min), (lat_max, lon_max) = bounds
    aspect_ratio = (lat_max - lat_min) / (lon_max - lon_min)
    width = 6
    height = width * aspect_ratio

    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    ax.axis('off')

    ax.imshow(
        grid,
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin='lower',
        vmin=vmin,
        vmax=vmax,
        cmap='Greys',
        alpha=0.9,
        aspect='auto'
    )

    # Bounding box of measured area
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
    print(f"✅ IDW label image saved to: {output_file}")

###############################################################################
def idw_loocv(
        target,
        df,
        k,
        power,
        output_file=""
):
    coords = np.vstack((df['longitude'].values, df['latitude'].values)).T
    trues = df[target].values

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

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("✅ IDW LOOCV completed.")
    print(f"RMSE: {rmse:.5f}, MAE: {mae:.5f}, R²: {r2:.3f}")

    if output_file:
        # Plot
        fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    
        # --- Scatter plot
        axs[0].scatter(y_true, y_pred, alpha=0.8)
        axs[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        axs[0].set_xlabel(f"True {target}")
        axs[0].set_ylabel(f"Predicted {target}")
        axs[0].set_title(f"IDW LOOCV - True vs Predicted\nRMSE={rmse:.5f}, MAE={mae:.5f}, R²={r2:.3f}")
        axs[0].grid(True)
        axs[0].axis("equal")
    
        # --- Line plot
        axs[1].plot(y_true, label="True", color="black", linewidth=1.5)
        axs[1].plot(y_pred, label="Predicted", color="blue", linestyle="--")
        axs[1].set_title("True vs Predicted (Index order)")
        axs[1].set_xlabel("Index")
        axs[1].set_ylabel(target)
        axs[1].grid(True)
        axs[1].legend()
    
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"✅ IDW LOOCV Image saved to: {output_file}")

    return rmse, mae, r2

###############################################################################
def save_idw_formula_as_jpg(filename="formula_idw.jpg"):
    formula = (
        r"$\hat{z}(x_0) = \frac{\sum_{i=1}^{k} w_i z_i}"
        r"{\sum_{i=1}^{k} w_i}, \quad \text{where } w_i = \frac{1}{d(x_0, x_i)^p}$"
    )

    explanation_lines = [
        r"$x_0$: location to interpolate",
        r"$x_i$: known data point location",
        r"$z_i$: known value at $x_i$",
        r"$d(x_0, x_i)$: distance between $x_0$ and $x_i$",
        r"$w_i$: weight of $z_i$",
        r"$p$: power parameter (controls weight decay)",
        r"$k$: number of nearest neighbors"
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')

    # Formula centered on top
    ax.text(0, 1, formula, fontsize=18, ha='left', va='center')

    # Explanation aligned to left below
    y_start = 0.7
    line_spacing = 0.07
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * line_spacing, line,
                fontsize=12, ha='left', va='center')

    plt.tight_layout()

    temp_file = "_temp_formula.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()

    os.remove(temp_file)
    print(f"✅ Saved JPEG formula image with explanation as {filename}")
