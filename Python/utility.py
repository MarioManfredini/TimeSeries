# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 2024

@author: Mario
"""
import os
import pandas as pd
import numpy as np
import glob
import logging

###############################################################################
# Constants
WD_COLUMN = 'WD(16Dir)'

###############################################################################
# Imposta il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

###############################################################################
def load_and_prepare_data(data_dir, prefecture_code, station_code):
    pattern = os.path.join(data_dir, f"*{prefecture_code}_{station_code}.csv")
    file_list = sorted(glob.glob(pattern))

    if not file_list:
        logger.error("No data files found for the specified station.")
        raise FileNotFoundError("No data files found for the specified station.")

    data_frames = []
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path, parse_dates=['日付'], encoding='shift_jis')
            data_frames.append(df)
            logger.info(f"Loaded: {os.path.basename(file_path)} with {len(df)} records.")
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

    data = pd.concat(data_frames, ignore_index=True)

    exclude_fields = ["測定局コード", "日付", "時"]
    potential_items = [col for col in data.columns if col not in exclude_fields]
    valid_items = []
    total_rows = len(data)

    wind_directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW",
        "CALM"
    ]

    for item in potential_items:
        if item == WD_COLUMN:
            valid_values = data[item].isin(wind_directions).sum()
            valid_ratio = valid_values / total_rows

            if valid_ratio >= 0.20:
                valid_items.append(item)
                logger.info(f"'{item}' retained ({valid_ratio:.1%} valid wind direction values).")
            else:
                data.drop(columns=[item], inplace=True)
                logger.warning(f"'{item}' dropped ({valid_ratio:.1%} valid wind directions below threshold).")
            continue

        data[item] = pd.to_numeric(data[item], errors='coerce')
        valid_count = data[item].notna().sum()
        valid_ratio = valid_count / total_rows

        if valid_ratio >= 0.20:
            valid_items.append(item)
            logger.info(f"'{item}' retained ({valid_ratio:.1%} valid numeric data).")
        else:
            data.drop(columns=[item], inplace=True)
            logger.warning(f"'{item}' dropped ({valid_ratio:.1%} valid numeric data below threshold).")

    # Conversione della colonna '日付' in datetime e set come indice
    data['datetime'] = pd.to_datetime(data.apply(adjust_datetime, axis=1), format='%Y-%m-%d %H:%M')
    data.set_index('datetime', inplace=True)

    # Pulizia righe con NaN solo sulle colonne numeriche
    numeric_items = [item for item in valid_items if item != WD_COLUMN]
    before_drop = len(data)
    data.dropna(subset=numeric_items, inplace=True)
    after_drop = len(data)
    logger.info(f"Cleaned data: removed {before_drop - after_drop} rows with missing numeric values.")

    if WD_COLUMN in valid_items:
        before_wd_drop = len(data)
        data = data[data[WD_COLUMN].isin(wind_directions)]
        after_wd_drop = len(data)
        logger.info(f"{WD_COLUMN} cleaned: removed {before_wd_drop - after_wd_drop} rows with invalid wind directions.")

    # Calcolo U/V se 'WD(16Dir)' è presente
    if WD_COLUMN in valid_items:
        data['WD_degrees'] = data[WD_COLUMN].map(direction_to_degrees)

        if 'WS(m/s)' in data.columns:
            data['WS(m/s)'] = pd.to_numeric(data['WS(m/s)'], errors='coerce')

        if 'WS(m/s)' in data.columns and 'WD_degrees' in data.columns:
            # Prepara le colonne U e V con zeri
            data['U'] = 0.0
            data['V'] = 0.0
        
            # Identifica i record validi con direzione non-NaN (quindi non 'CALM')
            valid_wind_mask = data['WD_degrees'].notna()
        
            # Calcola i componenti solo dove ha senso
            angle_rad = np.deg2rad(data.loc[valid_wind_mask, 'WD_degrees'])
            ws_values = data.loc[valid_wind_mask, 'WS(m/s)']
        
            data.loc[valid_wind_mask, 'U'] = ws_values * np.sin(angle_rad)
            data.loc[valid_wind_mask, 'V'] = ws_values * np.cos(angle_rad)
        
            logger.info("Wind components U and V calculated (including calm wind as 0).")


    data.sort_index(inplace=True)
    logger.info(f"Data prepared successfully with {len(data)} rows and {len(valid_items)} valid items.")

    return data, valid_items

###############################################################################
def adjust_datetime(row: pd.Series) -> str:
    if row['時'] == 24:
        new_date: pd.Timestamp = row['日付'] + pd.Timedelta(days=1)
        new_hour: str = '00'
    else:
        new_date = row['日付']
        new_hour = str(row['時']).zfill(2)
    return f"{new_date.strftime('%Y-%m-%d')} {new_hour}:00"


###############################################################################
# Conversion table from wind direction to degrees
wind_directions = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
    "CALM"
]

###############################################################################
# === Funzione per convertire direzione (stringa) in angolo gradi ===
def direction_to_degrees(direction):
    mapping = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90,
        "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180,
        "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270,
        "WNW": 292.5, "NW": 315, "NNW": 337.5, "CALM": np.nan
    }
    return mapping.get(direction, np.nan)


###############################################################################
# Conversion table from wind direction to degrees
wind_direction_degrees = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90,
    "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180,
    "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270,
    "WNW": 292.5, "NW": 315, "NNW": 337.5
}

###############################################################################
# Function to calculate U and V
def calculate_uv(row):
    direction = row[WD_COLUMN]
    speed = row['WS(m/s)']
    if direction == 'CALM' or pd.isna(direction) or pd.isna(speed):
        return pd.Series({'U': 0.0, 'V': 0.0})
    angle_deg = wind_direction_degrees.get(direction, np.nan)
    if np.isnan(angle_deg):
        return pd.Series({'U': np.nan, 'V': np.nan})
    angle_rad = np.deg2rad(angle_deg)
    u = speed * np.sin(angle_rad)
    v = speed * np.cos(angle_rad)
    return pd.Series({'U': u, 'V': v})
