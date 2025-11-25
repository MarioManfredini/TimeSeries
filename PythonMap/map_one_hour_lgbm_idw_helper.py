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
from PIL import Image

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
def save_lgbm_idw_formula_as_jpg(filename="formula_lgbm_idw.jpg"):
    """
    Save a visual explanation of the LightGBM + IDW formula as a JPEG image.

    Parameters:
        filename: output file path
    """

    formula = (
        r"$\hat{Z}(x) = \hat{Z}_{LGBM}(x) \;+\; "
        r"\frac{\sum_{i=1}^{N} \frac{\hat{Z}_{LGBM}(x_i)}{d(x,x_i)^p}}"
        r"{\sum_{i=1}^{N} \frac{1}{d(x,x_i)^p}}$"
    )

    explanation_lines = [
        r"$\hat{Z}(x)$ : Final estimated Ox at location $x$",
        r"$\hat{Z}_{LGBM}(x)$ : LightGBM prediction at location $x$",
        r"$x_i$ : Monitoring station locations",
        r"$d(x,x_i)$ : Distance between point $x$ and station $i$ (km)",
        r"$p$ : IDW power parameter (commonly $1 \sim 3$, here $p=2$)",
        r"The second term is the spatial interpolation (IDW) applied",
        r"to the LightGBM predictions to add spatial smoothness."
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis('off')

    # Title + formula
    ax.text(0.02, 0.95, "LightGBM + IDW Interpolation", fontsize=18,
            ha='left', va='center')
    ax.text(0.02, 0.75, formula, fontsize=20, ha='left', va='center')

    # Explanation lines
    y_start = 0.55
    line_spacing = 0.08
    for i, line in enumerate(explanation_lines):
        ax.text(0.02, y_start - i * line_spacing, line,
                fontsize=12, ha='left', va='center')

    plt.tight_layout()

    # Save temporary PNG
    temp_file = "_temp_lgbm_idw_formula.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

    # Convert to high-quality JPEG
    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved LGBM + IDW formula JPEG to {filename}")
