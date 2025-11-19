# -*- coding: utf-8 -*-
"""
Created: 2025-06-22
Author: Mario
"""

import pandas as pd
from geopy.geocoders import Nominatim
from time import sleep
import re

def normalize_address(raw_address):
    """
    Normalize a concatenated Japanese address for Nominatim.
    - Adds spaces between prefecture, city, ward, etc.
    - Removes house number
    Example:
      "大阪府大阪市此花区春日出北１－８－４"
    → "Japan 大阪府 大阪市 此花区 春日出北"
    """
    address = raw_address.strip()

    # Remove everything after the first house-number pattern
    address = re.sub(r'[０-９0-9一二三四五六七八九十百千万]+[－ー-][０-９0-9一二三四五六七八九十百千万－ー-]+$', '', address)

    # Add space after Japanese administrative division keywords
    # Split based on prefecture/city/ward/town/village identifiers
    keywords = ['都', '道', '府', '県', '市', '区', '町', '村']
    for kw in keywords:
        address = address.replace(kw, kw + ' ')

    # Remove multiple spaces
    address = re.sub(r'\s+', ' ', address)

    # Add "Japan" prefix
    return 'Japan ' + address.strip()

# === Config ===
input_file = 'OsakaStationAddress.csv'
output_file = 'OsakaStationAddress_Coordinates.csv'
wait_seconds = 2  # respect Nominatim rate limit

# === Initialize geocoder ===
geolocator = Nominatim(user_agent="ox-map-geocoder")

def geocode_address(address):
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"[ERROR] {address}: {e}")
    return None, None

# === Read input CSV file ===
df = pd.read_csv(input_file, encoding='utf-8-sig', skipinitialspace=True)

# Check required columns
required_columns = {'station_code', 'station_name', 'station_address'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"The file must contain the columns: {required_columns}")

# === Geocoding ===
results = []

for idx, row in df.iterrows():
    code = row['station_code']
    name = row['station_name']
    address = row['station_address']

    normalized = normalize_address(address)
    print(f"[{idx+1}/{len(df)}] Geocoding: {name} ({normalized})...")

    lat, lon = geocode_address(normalized)

    if lat is None or lon is None:
        print(f"  ⚠️ Coordinates not found for: {name}")
    else:
        print(f"  → Lat: {lat:.6f}, Lon: {lon:.6f}")

    results.append({
        'station_code': code,
        'station_name': name,
        'latitude': lat,
        'longitude': lon
    })

    sleep(wait_seconds)

# === Write output CSV ===
output_df = pd.DataFrame(results)
output_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✅ File saved as: {output_file}")
