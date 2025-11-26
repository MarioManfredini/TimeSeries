# -*- coding: utf-8 -*-
"""
Created: 2025/06/22

Author: Mario
"""

import os
import sys
import pandas as pd
import folium

from pathlib import Path
from datetime import datetime

from map_utils import generate_idw_images_by_hour
from anime_idw_helper import (
    inject_animation_and_controls
)

utility_dir = Path('..') / 'Python'
if utility_dir not in sys.path:
    sys.path.append(utility_dir)
import utility

###############################################################################
# === CONFIG ===
data_dir = Path('..') / 'data' / 'Osaka'
prefecture_code = '27'
prefecture_name = '大阪府'
station_coordinates = 'Stations_Ox.csv'
csv_path = os.path.join(data_dir, station_coordinates)
target = 'Ox(ppm)'
year = 2025
month = 5
day = 1
from_hour = 0
hours = 24

###############################################################################
stations_df = pd.read_csv(csv_path, skipinitialspace=True)
from_datetime = datetime(year, month, day, from_hour)

records = utility.load_ox_time_series(
    data_dir=data_dir,
    stations_df=stations_df,
    from_datetime=from_datetime,
    prefecture_code=prefecture_code,
    hours=hours
)

# === Folium Map ===
center_lat = sum(r['latitude'] for r in records) / len(records)
center_lon = sum(r['longitude'] for r in records) / len(records)
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# Save map
html_file = f'anime_idw_{hours}_hours_{prefecture_name}_{target}_from_{year}{month}{day}{from_hour}.html'
html_path = os.path.join("..", "html", html_file)
os.makedirs(os.path.dirname(html_path), exist_ok=True)
m.save(html_path)
print(f"Map saved to: {html_path}")

image_folder = Path('..') / 'html' / 'idw_frames'
image_prefix = "idw_"

bounds, vmin, vmax = generate_idw_images_by_hour(
    target,
    records,
    k=9,
    power=0.05,
    output_dir=image_folder,
    image_prefix=image_prefix,
    num_cells=500,
    overwrite=True
)

inject_animation_and_controls(
    target,
    html_file_path=html_path,
    image_folder=image_folder,
    bounds=bounds,
    vmin=vmin,
    vmax=vmax,
    image_prefix=image_prefix,
    interval_ms=1000
)
