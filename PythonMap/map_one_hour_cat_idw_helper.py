# -*- coding: utf-8 -*-
"""
Created: 2025/11/24
Author: Mario
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from PIL import Image

###############################################################################
def compute_cat_loocv_residuals(df, features, target='Ox(ppm)'):
    """
    LOOCV with CatBoostRegressor.
    """
    X = df[features].copy()
    y = df[target].values

    scaler = StandardScaler()
    X_scaled_df = pd.DataFrame(scaler.fit_transform(X), columns=features)

    model_params = dict(
        depth=8,
        iterations=500,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False,
        random_seed=42
    )

    preds = []
    for i in range(len(df)):
        X_train = X_scaled_df.drop(i)
        y_train = np.delete(y, i)
        X_test = X_scaled_df.iloc[[i]]

        model = CatBoostRegressor(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds.append(float(y_pred[0]))

    residuals_df = df[['station_code', 'latitude', 'longitude', target]].copy()
    residuals_df['prediction'] = preds
    residuals_df['residual'] = residuals_df[target] - residuals_df['prediction']

    print("✅ CatBoost LOOCV residuals computed.")
    return model, residuals_df, scaler

###############################################################################
def save_cat_idw_formula_as_jpg(filename="formula_cat_idw.jpg"):
    """
    Save a visual explanation of the CatBoost + IDW interpolation formula.
    """
    formula = (
        r"$\hat{Z}(x) = \hat{Z}_{CAT}(x) \;+\; "
        r"\frac{\sum_{i=1}^{N} \frac{\hat{Z}_{CAT}(x_i)}{d(x,x_i)^p}}"
        r"{\sum_{i=1}^{N} \frac{1}{d(x,x_i)^p}}$"
    )

    explanation_lines = [
        r"$\hat{Z}(x)$ : Final estimated Ox at location $x$",
        r"$\hat{Z}_{CAT}(x)$ : CatBoost prediction at location $x$",
        r"$x_i$ : Monitoring station locations",
        r"$d(x,x_i)$ : Distance between point $x$ and station $i$ (km)",
        r"$p$ : IDW power parameter (commonly $1 \sim 3$, here $p=2$)",
        r"IDW adds spatial smoothness to the CatBoost prediction surface."
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis('off')

    ax.text(0.02, 0.95, "CatBoost + IDW Interpolation", fontsize=18, ha='left')
    ax.text(0.02, 0.75, formula, fontsize=20, ha='left')

    y_start = 0.55
    line_spacing = 0.08
    for i, line in enumerate(explanation_lines):
        ax.text(0.02, y_start - i * line_spacing, line, fontsize=12, ha='left')

    plt.tight_layout()

    temp_file = "_temp_cat_idw_formula.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved CatBoost + IDW formula JPEG to {filename}")
