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
import re

from datetime import timedelta

###############################################################################
# Constants
WD_COLUMN = 'WD(16Dir)'

###############################################################################
# Imposta il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

###############################################################################
def load_and_prepare_data(data_dir, prefecture_code, station_code, remove_outliers: bool = False):
    file_list = _get_csv_file_list(data_dir, prefecture_code, station_code)
    data = _load_csv_files(file_list)
    valid_items = _filter_valid_columns(data)
    data = _set_datetime_index(data)
    numeric_items = [item for item in valid_items if item != WD_COLUMN]

    if remove_outliers:
        _remove_outliers(data, numeric_items)

    _handle_missing_values(data, numeric_items)
    data.dropna(subset=numeric_items, inplace=True)

    if WD_COLUMN in valid_items:
        data = _clean_and_process_wind(data)

    data.sort_index(inplace=True)
    logger.info(f"Data prepared successfully with {len(data)} rows and {len(valid_items)} valid items.")
    return data, valid_items

###############################################################################
def get_station_name(data_dir, station_code):
    """
    Returns the station name matching the given code from Stations_Ox.csv.
    Tries multiple encodings automatically and handles hidden spaces/BOM.
    """
    csv_path = os.path.join(data_dir, 'Stations_Ox.csv')
    encodings = ['cp932', 'shift_jis', 'utf-8-sig', 'utf-8']

    for enc in encodings:
        try:
            df = pd.read_csv(
                csv_path,
                encoding=enc,
                dtype=str,
                skipinitialspace=True  # remove spaces after commas
            )

            # Remove hidden BOM and trim whitespace
            df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)
            df['station_code'] = df['station_code'].str.strip().str.replace('\ufeff', '', regex=False)
            df['station_name'] = df['station_name'].str.strip()

            if 'station_code' not in df.columns:
                logger.warning(f"Encoding '{enc}': column 'station_code' not found → columns = {df.columns.tolist()}")
                continue

            logger.info(f"Encoding '{enc}' OK → {len(df)} rows loaded")

            match = df[df['station_code'] == str(station_code)]
            if not match.empty:
                station_name = match.iloc[0]['station_name']
                logger.info(f"Found station_name = {station_name}")
                return station_name
            else:
                logger.warning(f"Code '{station_code}' not found in data (encoding: {enc})")

        except FileNotFoundError:
            logger.error(f"File Stations_Ox.csv not found in {data_dir}")
            return None
        except UnicodeDecodeError:
            logger.info(f"Encoding [{enc}] failed. Try next encoding.")
            continue
        except Exception as e:
            logger.error(f"Error reading Stations_Ox.csv with encoding '{enc}': {e}")
            return None

    logger.error("Unable to read Stations_Ox.csv with any known encoding.")
    return None

###############################################################################
def _get_csv_file_list(data_dir, prefecture_code, station_code):
    pattern = os.path.join(data_dir, f"*{prefecture_code}_{station_code}.csv")
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        logger.error("No data files found for the specified station.")
        raise FileNotFoundError("No data files found for the specified station.")
    return file_list

###############################################################################
def _load_csv_files(file_list):
    data_frames = []
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path, parse_dates=['日付'], encoding='shift_jis')
            data_frames.append(df)
            logger.info(f"Loaded: {os.path.basename(file_path)} with {len(df)} records.")
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
    return pd.concat(data_frames, ignore_index=True)

###############################################################################
def _filter_valid_columns(data):
    exclude_fields = ["測定局コード", "日付", "時"]
    potential_items = [col for col in data.columns if col not in exclude_fields]
    valid_items = []
    total_rows = len(data)

    wind_directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "CALM"
    ]

    for item in potential_items:
        if item == WD_COLUMN:
            valid_ratio = data[item].isin(wind_directions).sum() / total_rows
        else:
            data[item] = pd.to_numeric(data[item], errors='coerce')
            valid_ratio = data[item].notna().sum() / total_rows

        if valid_ratio >= 0.20:
            valid_items.append(item)
            logger.info(f"'{item}' retained ({valid_ratio:.1%} valid values).")
        else:
            data.drop(columns=[item], inplace=True)
            logger.warning(f"'{item}' dropped ({valid_ratio:.1%} valid values below threshold).")

    return valid_items

###############################################################################
def _adjust_datetime(row: pd.Series) -> str:
    if row['時'] == 24:
        new_date: pd.Timestamp = row['日付'] + pd.Timedelta(days=1)
        new_hour: str = '00'
    else:
        new_date = row['日付']
        new_hour = str(row['時']).zfill(2)
    return f"{new_date.strftime('%Y-%m-%d')} {new_hour}:00"

def _set_datetime_index(data):
    data['datetime'] = pd.to_datetime(data.apply(_adjust_datetime, axis=1), format='%Y-%m-%d %H:%M')
    data.set_index('datetime', inplace=True)
    return data

###############################################################################
def _remove_outliers(data, numeric_items):
    for col in numeric_items:
        z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
        outliers = z_scores > 3
        if outliers.any():
            logger.info(f"Outlier detected in '{col}': replacing {outliers.sum()} values.")
            data.loc[outliers, col] = np.nan

###############################################################################
def _handle_missing_values(data, numeric_items):
    for col in numeric_items:
        before_nan = data[col].isna().sum()
        data[col] = data[col].interpolate(method='time', limit_direction='both')
        after_nan = data[col].isna().sum()
        logger.info(f"'{col}': {before_nan} NaNs -> {after_nan} after interpolation.")

        if after_nan > 0:
            data[col].fillna(data[col].mean(), inplace=True)
            logger.info(f"'{col}': remaining NaNs filled with column mean.")

###############################################################################
def _direction_to_degrees(direction):
    mapping = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90,
        "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180,
        "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270,
        "WNW": 292.5, "NW": 315, "NNW": 337.5, "CALM": np.nan
    }
    return mapping.get(direction, np.nan)

###############################################################################
def _clean_and_process_wind(data):
    wind_directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "CALM"
    ]
    before = len(data)
    data = data[data[WD_COLUMN].isin(wind_directions)].copy()
    after = len(data)
    logger.info(f"{WD_COLUMN} cleaned: removed {before - after} rows with invalid wind directions.")

    if 'WS(m/s)' in data.columns:
        data['WS(m/s)'] = pd.to_numeric(data['WS(m/s)'], errors='coerce')
        data['WD_degrees'] = data[WD_COLUMN].map(_direction_to_degrees)
        valid_mask = data['WD_degrees'].notna()
        angle_rad = np.deg2rad(data.loc[valid_mask, 'WD_degrees'])
        ws_values = data.loc[valid_mask, 'WS(m/s)']
        data['U'] = 0.0
        data['V'] = 0.0
        data.loc[valid_mask, 'U'] = ws_values * np.sin(angle_rad)
        data.loc[valid_mask, 'V'] = ws_values * np.cos(angle_rad)
        logger.info("Wind components U and V calculated.")
    
    return data

###############################################################################
def load_ox_time_series(data_dir, stations_df, from_datetime, prefecture_code, hours=48):
    """
    Estrae i valori di Ox, WS, WD per le ultime `hours` ore da `from_datetime`
    """
    to_datetime = from_datetime + timedelta(hours=hours)
    all_data = []

    for _, row in stations_df.iterrows():
        station_code = row['station_code']
        station_name = row['station_name']
        lat = row['latitude']
        lon = row['longitude']

        # Determina gli anni e i mesi necessari
        months_needed = pd.date_range(from_datetime, to_datetime, freq='MS').to_period('M').unique()
        for period in months_needed:
            ym_str = f"{period.year:04d}{period.month:02d}"
            filename = f"{ym_str}_{prefecture_code}_{station_code}.csv"
            file_path = os.path.join(data_dir, filename)

            if not os.path.exists(file_path):
                print(f"[WARN] File mancante: {file_path}")
                continue

            try:
                df = pd.read_csv(file_path, encoding='cp932')
                df['datetime'] = pd.to_datetime(
                    df['日付'] + ' ' + df['時'].astype(str).str.zfill(2),
                    format='%Y/%m/%d %H',
                    errors='coerce'
                )
                df = df.dropna(subset=['datetime'])
                df = df[(df['datetime'] >= from_datetime) & (df['datetime'] < to_datetime)]

                # Pulisce e converte i valori
                df['Ox(ppm)'] = pd.to_numeric(df['Ox(ppm)'], errors='coerce')
                df['WS(m/s)'] = pd.to_numeric(df['WS(m/s)'], errors='coerce')
                df = df[df['Ox(ppm)'].notna()]

                df['WD_degrees'] = df['WD(16Dir)'].map(_direction_to_degrees)
                df['U'] = df['WS(m/s)'] * np.sin(np.deg2rad(df['WD_degrees']))
                df['V'] = df['WS(m/s)'] * np.cos(np.deg2rad(df['WD_degrees']))

                for _, r in df.iterrows():
                    all_data.append({
                        'station_code': station_code,
                        'station_name': station_name,
                        'latitude': lat,
                        'longitude': lon,
                        'datetime': r['datetime'],
                        'Ox(ppm)': r['Ox(ppm)'],
                        'WS(m/s)': r['WS(m/s)'],
                        'WD(16Dir)': r['WD(16Dir)'],
                        'U': r['U'],
                        'V': r['V']
                    })

            except Exception as e:
                print(f"[ERROR] Errore nel file {file_path}: {e}")

    return all_data

###############################################################################
def sanitize_filename_component(text: str) -> str:
    """
    Sanitize a string for safe use in filenames (Windows/Linux).
    Keeps scientific readability.
    """
    replacements = {
        '/': '_per_',
        'μ': 'u',
        '℃': 'C',
        '％': 'percent',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove any remaining unsafe characters
    text = re.sub(r'[<>:"\\|?*]', '', text)

    return text
