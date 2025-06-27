# -*- coding: utf-8 -*-
"""
Created: 2025/06/22

Author: Mario
"""

import os
import sys
import pandas as pd
import folium
from datetime import datetime
from folium.plugins import TimestampedGeoJson
from map_utils import create_geojson_from_records

utility_dir = os.path.abspath('..\\Python\\')
if utility_dir not in sys.path:
    sys.path.append(utility_dir)
import utility

# === CONFIG ===
data_dir = '..\\data\\Osaka\\'
prefecture_code = '27'
station_coordinates = 'Stations_Ox.csv'
csv_path = os.path.join(data_dir, station_coordinates)

stations_df = pd.read_csv(csv_path, skipinitialspace=True)
from_datetime = datetime(2025, 5, 1, 0)

records = utility.load_ox_time_series(
    data_dir=data_dir,
    stations_df=stations_df,
    from_datetime=from_datetime,
    prefecture_code=prefecture_code,
    hours=48
)

print(f"{len(records)} total hourly records loaded.")

# === Folium Map ===
center_lat = sum(r['latitude'] for r in records) / len(records)
center_lon = sum(r['longitude'] for r in records) / len(records)
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

geojson_data = create_geojson_from_records(records)

TimestampedGeoJson(
    data=geojson_data,
    transition_time=200,
    loop=False,
    auto_play=False,
    add_last_point=True,
    period="PT1H"
).add_to(m)

# Save interactive map
m.save("OxMapAnimation_LastTwoDays.html")
print("Map saved to: OxMapAnimation_LastTwoDays.html")
