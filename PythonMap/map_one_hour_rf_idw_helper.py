# -*- coding: utf-8 -*-
"""
Created: 2025/11/24
Author: Mario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

###############################################################################
def compute_rf_loocv_residuals(df, features, target='Ox(ppm)'):

    X = df[features]
    y = df[target].values

    scaler = StandardScaler()
    X_scaled_df = pd.DataFrame(scaler.fit_transform(X), columns=features)

    preds = []
    trues = []

    for i in range(len(df)):
        X_train = X_scaled_df.drop(i)
        y_train = np.delete(y, i)
        X_test = X_scaled_df.iloc[[i]]

        model = RandomForestRegressor(
            n_estimators=550,
            max_depth=5,
            min_samples_leaf=1,
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
def save_rf_idw_formula_as_jpg(filename="formula_rf_idw.jpg"):

    fig, ax = plt.subplots(figsize=(6, 2), dpi=200)
    ax.axis("off")

    text = (
        r"$\hat{Z}(x) = \hat{Z}_{RF}(x) \;+\; "
        r"\frac{\sum_{i=1}^{N} \frac{\hat{Z}_{RF}(x_i)}{d(x,x_i)^p}}"
        r"{\sum_{i=1}^{N} \frac{1}{d(x,x_i)^p}}$"
    )

    ax.text(0.5, 0.5, text, fontsize=14, ha="center", va="center")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.2, dpi=200)
    plt.close()
