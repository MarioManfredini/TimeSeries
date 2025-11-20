# -*- coding: utf-8 -*-
"""
Created: 2025/06/29
Author: Mario
Description: Spatial LightGBM with LOOCV, map rendering, and PDF report.
"""
import os
import pandas as pd
import numpy as np
import folium

from pathlib import Path
from folium.raster_layers import ImageOverlay
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from map_utils import (
    load_station_temporal_features,
    compute_bounds_from_df
)
from map_report import (
    capture_html_map_screenshot,
    save_model_report_pdf, plot_loocv_results
)
from map_one_hour_lgbm_helper import (
    generate_image_with_labels_only,
    generate_rf_confidence_overlay_image,
    save_rf_formula_as_jpg
)

###############################################################################
# === CONFIG ===
data_dir = Path('..') / 'data' / 'Osaka'
prefecture_code = '27'
prefecture_name = '大阪府'
station_coordinates = 'Stations_Ox.csv'
csv_path = os.path.join(data_dir, station_coordinates)
target = 'Ox(ppm)'
year=2025
month=5
day=12
hour=19

# === Load and prepare data ===
df = pd.read_csv(csv_path, skipinitialspace=True)

df = load_station_temporal_features(
    data_dir,
    df,
    prefecture_code,
    year,
    month,
    day,
    hour,
    lags=24,
    target_item=target
)

print("[DEBUG] Final dataframe shape:", df.shape)

# Exclude non-numeric columns like 'datetime' and 'WD(16Dir)'
exclude_cols = ['datetime', 'WD(16Dir)', 'station_code', 'station_name'] 
features = [col for col in df.columns if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)]

X = df[features]
y = df[target].values

# === Scaler initialization ===
scaler = StandardScaler()
X_scaled_df = pd.DataFrame(scaler.fit_transform(X), columns=features)

preds, trues = [], []
for i in range(len(df)):
    X_train = X_scaled_df.drop(i)
    y_train = np.delete(y, i)
    X_test = X_scaled_df.iloc[[i]]

    model = LGBMRegressor(
        objective='regression',
        boosting_type='gbdt',
        n_estimators=400,
        learning_rate=0.02,
        max_depth=-1,
        random_state=42, # this number is used to seed the C++ code
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    preds.append(y_pred[0])
    trues.append(y[i])

mse = mean_squared_error(trues, preds)
rmse = mse ** 0.5
mae = mean_absolute_error(trues, preds)
r2 = r2_score(trues, preds)
print("\n✅ LightGBM LOOCV:")
print(f"RMSE: {rmse:.5f}, MAE: {mae:.5f}, R²: {r2:.3f}")

# === LOOCV ===
loocv_image="loocv.png"
loocv_image_path = os.path.join(".", "tmp", loocv_image)
os.makedirs(os.path.dirname(loocv_image_path), exist_ok=True)
plot_loocv_results(target, rmse, mae, r2, trues, preds, loocv_image_path)

# === Generate image ===
bounds = compute_bounds_from_df(df, margin_ratio=0.10)
(lat_min, lon_min), (lat_max, lon_max) = bounds
vmin = np.percentile(df[target], 5)
vmax = np.percentile(df[target], 95)

output_file = 'prediction.png'
output_file_path = os.path.join(".", "tmp", output_file)
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

generate_rf_confidence_overlay_image(
    df,
    bounds,
    model,
    features,
    vmin,
    vmax,
    output_file=output_file_path,
    num_cells=300,
    max_distance_km=15,
    cmap_name="Reds",
    scaler=scaler
)

# === Folium Map ===
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
image_overlay = ImageOverlay(
    name=f"LGBM Predicted {target}",
    image=output_file_path,
    bounds=[[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]],
    opacity=0.7
)
image_overlay.add_to(m)

html_file = (
    f'map_lgbm_one_hour_{prefecture_name}_{target}_{year}{month}{day}{hour}.html'
)
html_path = os.path.join("..", "html", html_file)
os.makedirs(os.path.dirname(html_path), exist_ok=True)
m.save(html_path)
print(f"✅ Map saved to: {html_path}")

formula_image_path = 'formula.jpg'
formula_image = 'formula.jpg'
formula_image_path = os.path.join(".", "tmp", formula_image)
os.makedirs(os.path.dirname(formula_image_path), exist_ok=True)
save_rf_formula_as_jpg(formula_image_path)

labels_image = 'labels_image.png'
labels_image_path = os.path.join(".", "tmp", labels_image)
os.makedirs(os.path.dirname(labels_image_path), exist_ok=True)
generate_image_with_labels_only(
    df,
    bounds=(lat_min, lon_min, lat_max, lon_max),
    output_file=labels_image_path,
)

screenshot = "screenshot.jpg"
screenshot_path = os.path.join(".", "tmp", screenshot)
os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
capture_html_map_screenshot(html_path, screenshot_path)

# === Report PDF ===
results=[(rmse, mae, r2)]
column_headers = ["RMSE", "MAE", "R²"]
table_data = []
for rmse, mae, r2 in results:
    table_data.append([
        f"{rmse:.5f}",
        f"{mae:.5f}",
        f"{r2:.3f}"
    ])

###############################################################################
# === Save Report ===
report_file = (
    f'map_lgbm_{prefecture_name}_{target}_{year}{month:02}{day:02}_{hour:02}00.pdf'
)
report_path = os.path.join("..", "reports", report_file)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

save_model_report_pdf(
    output_path=report_path,
    table_data=table_data,
    column_headers=column_headers,
    formula_image_path=formula_image_path,
    map_image_path=screenshot_path,
    labels_image_path=labels_image_path,
    additional_image_path=loocv_image_path,
    title=f"LightGBM Interpolation - {prefecture_name} - {year}/{month}/{day} {hour:02d}H"
)
