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
    Normalizza un indirizzo giapponese concatenato per Nominatim.
    - Aggiunge spazi tra prefettura, città, quartiere
    - Rimuove numero civico
    Esempio:
      "大阪府大阪市此花区春日出北１－８－４"
    → "日本 大阪府 大阪市 此花区 春日出北"
    """
    address = raw_address.strip()

    # Rimuove tutto dopo il primo numero civico
    address = re.sub(r'[０-９0-9一二三四五六七八九十百千万]+[－ー-][０-９0-9一二三四五六七八九十百千万－ー-]+$', '', address)

    # Aggiunge spazio dopo i termini giapponesi geografici
    # 都道府県市区町村を基に分割
    keywords = ['都', '道', '府', '県', '市', '区', '町', '村']
    for kw in keywords:
        address = address.replace(kw, kw + ' ')

    # Rimuove spazi multipli
    address = re.sub(r'\s+', ' ', address)

    # Aggiunge prefisso "日本"
    return '日本 ' + address.strip()

# === Config ===
input_file = 'OsakaStationAddress.csv'
output_file = 'OsakaStationAddress_Coordinates.csv'
wait_seconds = 2  # rispetta il rate limit di Nominatim

# === Inizializza geocoder ===
geolocator = Nominatim(user_agent="ox-map-geocoder")

def geocode_address(address):
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"[ERROR] {address}: {e}")
    return None, None

# === Leggi file CSV input ===
df = pd.read_csv(input_file, encoding='utf-8-sig', skipinitialspace=True)

# Verifica colonne richieste
required_columns = {'station_code', 'station_name', 'station_address'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Il file deve contenere le colonne: {required_columns}")

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
        print(f"  ⚠️ Coordinate non trovate per: {name}")
    else:
        print(f"  → Lat: {lat:.6f}, Lon: {lon:.6f}")

    results.append({
        'station_code': code,
        'station_name': name,
        'latitude': lat,
        'longitude': lon
    })

    sleep(wait_seconds)

# === Scrivi il CSV di output ===
output_df = pd.DataFrame(results)
output_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✅ File salvato come: {output_file}")