# -*- coding: utf-8 -*-
"""
Created: 2025/06/22

Author: Mario
Description: Visualize latest measurements on a map using grayscale intensity.
"""

import pandas as pd
import folium
import os

from pathlib import Path

###############################################################################
# File and path configuration
data_dir = Path('..') / 'data' / 'Osaka'
prefecture_code = '27'
prefecture_name = '大阪府'
target = 'Ox(ppm)'
station_coordinates = 'Stations_Ox.csv'
csv_path = os.path.join(data_dir, station_coordinates)
year = 2025
month = 5

###############################################################################
def load_latest_values(target, data_dir, stations_df, year, month, prefecture_code):
    """
    For each station, load the latest available value
    from the corresponding CSV file in the folder.
    """
    data = {}

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

            # Convert valuse to float and drop missing values
            df[target] = pd.to_numeric(df[target], errors='coerce')
            df = df[df[target].notna()]

            # Create datetime column from date and hour
            df["datetime"] = pd.to_datetime(
                df["日付"] + " " + df["時"].astype(str).str.zfill(2),
                errors='coerce',
                format='%Y/%m/%d %H'
            )

            # Drop invalid datetimes and sort
            df = df.dropna(subset=["datetime"])
            df = df.sort_values("datetime")

            # Take the last available value
            last_value = df[target].iloc[-1]
            data[station_code] = float(last_value)

        except Exception as e:
            print(f"[ERROR] Error reading file {file_path}: {e}")

    return data

###############################################################################
# Load station coordinates
df = pd.read_csv(csv_path, skipinitialspace=True)

# Load latest values
data = load_latest_values(
    target,
    data_dir,
    stations_df=df,
    year=year,
    month=month,
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

# Convert value to grayscale (lighter = higher value)
def get_gray_shade(value, vmin=0.0, vmax=0.1):
    norm_value = max(min(float(value), vmax), vmin)
    scale = int(255 * (1 - (norm_value - vmin) / (vmax - vmin)))
    return f'rgb({scale},{scale},{scale})'

# Add each station as a circle marker
for _, row in df.iterrows():
    code = row['station_code']
    value = data.get(code)

    fill_color = get_gray_shade(value) if value is not None else 'white'

    popup_text = (
        f"{row['station_name']} ({code})\n{target}: {value:.3f}"
        if value is not None else
        f"{row['station_name']} ({code})\n{target}: N/A"
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
html_file = (
    f'map_one_hour_{prefecture_name}_{target}_{year}{month}_lasthour.html'
)
html_path = os.path.join("..", "html", html_file)
os.makedirs(os.path.dirname(html_path), exist_ok=True)
m.save(html_path)
print(f"Map saved to: {html_path}")
