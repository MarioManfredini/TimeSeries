# -*- coding: utf-8 -*-
"""
Created: 2025/06/22

Author: Mario
Description: Visualize latest Ox(ppm) measurements on a map using grayscale intensity.
"""

import pandas as pd
import folium
import os

###############################################################################
def load_latest_ox_values(data_dir, stations_df, year, month, prefecture_code):
    """
    For each station, load the latest available Ox(ppm) value
    from the corresponding CSV file in the folder.
    """
    ox_data = {}

    # Format year and month as YYYYMM
    ym_str = f"{year:04d}{month:02d}"

    for _, row in stations_df.iterrows():
        station_code = row['station_code']
        filename = f"{ym_str}_{prefecture_code}_{station_code}.csv"
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            print(f"[WARN] File not found: {file_path}")
            continue

        try:
            # Load CSV with Japanese encoding
            df = pd.read_csv(file_path, encoding='cp932')

            # Convert Ox(ppm) to float and drop missing values
            df["Ox(ppm)"] = pd.to_numeric(df["Ox(ppm)"], errors='coerce')
            df = df[df["Ox(ppm)"].notna()]

            # Create datetime column from date and hour
            df["datetime"] = pd.to_datetime(
                df["日付"] + " " + df["時"].astype(str).str.zfill(2),
                errors='coerce',
                format='%Y/%m/%d %H'
            )

            # Drop invalid datetimes and sort
            df = df.dropna(subset=["datetime"])
            df = df.sort_values("datetime")

            # Take the last available Ox value
            last_ox = df["Ox(ppm)"].iloc[-1]
            ox_data[station_code] = float(last_ox)

        except Exception as e:
            print(f"[ERROR] Error reading file {file_path}: {e}")

    return ox_data

###############################################################################
# File and path configuration
data_dir = '..\\data\\Osaka\\'
prefecture_code = '27'
station_coordinates = 'Stations_Ox.csv'
csv_path = os.path.join(data_dir, station_coordinates)

# Load station coordinates
df = pd.read_csv(csv_path, skipinitialspace=True)

# Load latest Ox(ppm) values
ox_data = load_latest_ox_values(
    data_dir=data_dir,
    stations_df=df,
    year=2025,
    month=5,
    prefecture_code=prefecture_code
)

# Ensure required columns exist
required_columns = {'station_code', 'station_name', 'latitude', 'longitude'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

# Create base map centered on average coordinates
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# Convert Ox(ppm) to grayscale (lighter = higher value)
def get_gray_shade(ox_value, min_ox=0.0, max_ox=0.1):
    value = max(min(float(ox_value), max_ox), min_ox)
    scale = int(255 * (1 - (value - min_ox) / (max_ox - min_ox)))
    return f'rgb({scale},{scale},{scale})'

# Add each station as a circle marker
for _, row in df.iterrows():
    code = row['station_code']
    ox_value = ox_data.get(code)

    fill_color = get_gray_shade(ox_value) if ox_value is not None else 'white'

    popup_text = (
        f"{row['station_name']} ({code})\nOx: {ox_value:.3f} ppm"
        if ox_value is not None else
        f"{row['station_name']} ({code})\nOx: N/A"
    )

    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=6,
        popup=popup_text,
        color='black',
        fill=True,
        fill_color=fill_color,
        fill_opacity=0.8
    ).add_to(m)

# Save the map as an HTML file
output_path = 'OxMap_LastHour.html'
m.save(output_path)
print(f"[INFO] Map saved to: {output_path}")
