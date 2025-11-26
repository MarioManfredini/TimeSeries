# -*- coding: utf-8 -*-
"""
Created: 2025/11/24
Author: Mario
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from PIL import Image

###############################################################################
def compute_xgb_loocv_residuals(df, features, target='Ox(ppm)'):
    """
    LOOCV with XGBoost: train XGBRegressor n times leaving one station out,
    collect LOOCV predictions and residuals.

    Returns: (last_model, residuals_df, scaler)
    residuals_df columns: ['station_code', 'latitude', 'longitude', target, 'prediction', 'residual']
    """
    X = df[features].copy()
    y = df[target].values

    scaler = StandardScaler()
    X_scaled_df = pd.DataFrame(scaler.fit_transform(X), columns=features)

    # XGB hyperparameters - tweak as needed
    model_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    preds = []
    for i in range(len(df)):
        X_train = X_scaled_df.drop(i)
        y_train = np.delete(y, i)
        X_test = X_scaled_df.iloc[[i]]

        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds.append(float(y_pred[0]))

    residuals_df = df[['station_code', 'latitude', 'longitude', target]].copy()
    residuals_df['prediction'] = preds
    residuals_df['residual'] = residuals_df[target] - residuals_df['prediction']

    print("✅ XGBoost LOOCV residuals computed.")
    return model, residuals_df, scaler

###############################################################################
def save_xgb_idw_formula_as_jpg(filename="formula_xgb_idw.jpg"):
    """
    Save a visual explanation of the XGBoost + IDW formula as a JPEG image.
    """
    formula = (
        r"$\hat{Z}(x) = \hat{Z}_{XGB}(x) \;+\; "
        r"\frac{\sum_{i=1}^{N} \frac{\hat{Z}_{XGB}(x_i)}{d(x,x_i)^p}}"
        r"{\sum_{i=1}^{N} \frac{1}{d(x,x_i)^p}}$"
    )
    explanation_lines = [
        r"$\hat{Z}(x)$ : Final estimated Ox at location $x$",
        r"$\hat{Z}_{XGB}(x)$ : XGBoost prediction at location $x$",
        r"$x_i$ : Monitoring station locations",
        r"$d(x,x_i)$ : Distance between point $x$ and station $i$ (km)",
        r"$p$ : IDW power parameter (commonly $1 \sim 3$, here $p=2$)",
        r"The second term is the spatial interpolation (IDW) applied",
        r"to the XGBoost predictions to add spatial smoothness."
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis('off')

    ax.text(0.02, 0.95, "XGBoost + IDW Interpolation", fontsize=18, ha='left', va='center')
    ax.text(0.02, 0.75, formula, fontsize=20, ha='left', va='center')

    y_start = 0.55
    line_spacing = 0.08
    for i, line in enumerate(explanation_lines):
        ax.text(0.02, y_start - i * line_spacing, line, fontsize=12, ha='left', va='center')

    plt.tight_layout()

    temp_file = "_temp_xgb_idw_formula.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)
    print(f"✅ Saved XGB + IDW formula JPEG to {filename}")
