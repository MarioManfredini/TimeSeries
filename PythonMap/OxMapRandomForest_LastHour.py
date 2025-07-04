# -*- coding: utf-8 -*-
"""
Created: 2025/06/29
Author: Mario
Description: Spatial Random Forest with LOOCV, map rendering, and PDF report.
"""
import os
import pandas as pd
import numpy as np
import folium
from folium.raster_layers import ImageOverlay
from map_utils import load_latest_ox_values, compute_wind_uv, compute_bounds_from_df, generate_rf_prediction_image
from map_report import capture_html_map_screenshot, save_rf_report_pdf, save_rf_formula_as_jpg
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === CONFIG ===
data_dir = '..\\data\\Osaka\\'
prefecture_code = '27'
station_coordinates = 'Stations_Ox.csv'
csv_path = os.path.join(data_dir, station_coordinates)

# === Load and prepare data ===
# Filtra solo le stazioni presenti nel file CSV con coordinate
df = pd.read_csv(csv_path, skipinitialspace=True)

# Carica dati da file CSV mensile
ox_data = load_latest_ox_values(data_dir, df, year=2025, month=5, prefecture_code=prefecture_code)

# Mantieni solo stazioni per cui sono disponibili dati
df = df[df['station_code'].isin(ox_data.keys())].copy()

# Aggiungi tutte le colonne da ox_data
required_keys = ['Ox(ppm)', 'WS(m/s)', 'WD(16Dir)', 'NO(ppm)', 'NO2(ppm)']
for key in required_keys:
    df[key] = df['station_code'].map(lambda c: ox_data[c].get(key, np.nan))

# Calcolo del vento e rimozione dei NaN
df['U'], df['V'] = compute_wind_uv(df['WS(m/s)'], df['WD(16Dir)'])
df = df.dropna(subset=['Ox(ppm)', 'NO(ppm)', 'NO2(ppm)', 'U', 'V'])


# === Features and LOOCV ===
features = ['longitude', 'latitude', 'NO(ppm)', 'NO2(ppm)', 'U', 'V']
target = 'Ox(ppm)'
X = df[features].values
y = df[target].values

preds, trues = [], []
for i in range(len(df)):
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i)
    X_test = X[i].reshape(1, -1)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    preds.append(y_pred[0])
    trues.append(y[i])

mse = mean_squared_error(trues, preds)
rmse = mse ** 0.5
mae = mean_absolute_error(trues, preds)
r2 = r2_score(trues, preds)
print("\n✅ Random Forest LOOCV:")
print(f"RMSE: {rmse:.5f}, MAE: {mae:.5f}, R²: {r2:.3f}")

# === Generate image ===
bounds = compute_bounds_from_df(df, margin_ratio=0.10)
ox_min = np.percentile(df['Ox(ppm)'], 5)
ox_max = np.percentile(df['Ox(ppm)'], 95)
rf_image_path = "ox_rf_prediction.png"
generate_rf_prediction_image(
    df,
    bounds,
    model,
    features,
    ox_min,
    ox_max,
    output_file=rf_image_path,
    num_cells=300,
    cmap_name="Reds"
)

# === Folium Map ===
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
image_overlay = ImageOverlay(
    name="RF Predicted Ox(ppm)",
    image=rf_image_path,
    bounds=[[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]],
    opacity=0.7
)
image_overlay.add_to(m)
m.save("OxMapRF_LastHour.html")
print("Map saved to: OxMapRF_LastHour.html")

# === Report PDF ===
screenshot_path = "map_rf_screenshot.jpg"
capture_html_map_screenshot("OxMapRF_LastHour.html", screenshot_path)
save_rf_formula_as_jpg("formula_rf.jpg")
save_rf_report_pdf(
    output_path="RF_Report.pdf",
    results=[(rmse, mae, r2, "All stations (t = now)")],
    formula_image_path="formula_rf.jpg",
    html_screenshot_path=screenshot_path,
    rf_labels_image_path=rf_image_path,
    title="Random Forest Spatial Interpolation Report"
)

