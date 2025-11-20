# -*- coding: utf-8 -*-
"""
Created: 2025/06/22

Author: Mario
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from collections import defaultdict
from matplotlib.patches import Rectangle
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from datetime import datetime

###############################################################################
def load_hourly_measurements(
    data_dir,
    stations_df,
    prefecture_code,
    year,
    month,
    day=None,
    hour=None,
    measurement_columns=["Ox(ppm)"],
    include_wind=True
):
    """
    Load specified pollutant measurements (e.g. Ox, SO2, SPM) for each station for a given date and hour.
    Optionally include wind speed/direction if present.

    Parameters:
        data_dir (str): Directory where CSV files are stored.
        stations_df (pd.DataFrame): DataFrame with 'station_code' column.
        prefecture_code (str): Prefecture identifier used in file names.
        year, month (int): Year and month of data to load.
        day, hour (int, optional): Specific day and hour to filter. If None, use latest available data.
        measurement_columns (list of str): List of measurement column names to load (e.g., ["Ox(ppm)", "SO2(ppm)"]).
        include_wind (bool): Whether to load WS(m/s) and WD(16Dir) if present.

    Returns:
        dict: Dictionary keyed by station_code with a dict of requested measurements.
    """
    results = {}
    ym_str = f"{year:04d}{month:02d}"

    for _, row in stations_df.iterrows():
        station_code = row["station_code"]
        filename = f"{ym_str}_{prefecture_code}_{station_code}.csv"
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            print(f"[WARN] File not found: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path, encoding='cp932')

            # Parse datetime column
            df["datetime"] = pd.to_datetime(
                df["日付"] + " " + df["時"].astype(str).str.zfill(2),
                format='%Y/%m/%d %H',
                errors='coerce'
            )
            df = df.dropna(subset=["datetime"])

            # Use specific datetime or latest available
            if day is not None and hour is not None:
                dt_target = datetime(year, month, day, hour)
                row_match = df[df["datetime"] == dt_target]
                if row_match.empty:
                    print(f"[WARN] No data for {dt_target} at station {station_code}")
                    continue
                target_row = row_match.iloc[0]
            else:
                df = df.sort_values("datetime")
                target_row = df.iloc[-1]

            # Extract measurements
            station_data = {}
            for col in measurement_columns:
                if col in df.columns:
                    val = pd.to_numeric(target_row[col], errors='coerce')
                    station_data[col] = float(val) if pd.notna(val) else np.nan
                else:
                    station_data[col] = np.nan
                    print(f"[INFO] Column '{col}' not found in {filename}, setting NaN.")

            # Wind speed/direction if required
            if include_wind:
                ws = pd.to_numeric(target_row.get("WS(m/s)", np.nan), errors='coerce')
                wd = target_row.get("WD(16Dir)", np.nan)
                station_data["WS(m/s)"] = float(ws) if pd.notna(ws) else np.nan
                station_data["WD(16Dir)"] = wd if pd.notna(wd) else np.nan

            results[station_code] = station_data

        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")

    return results

###############################################################################
def load_preprocessed_hourly_data(
    data_dir: str,
    station_csv: str,
    prefecture_code: str,
    year: int,
    month: int,
    day: int,
    hour: int,
    target: str = 'Ox(ppm)'
) -> pd.DataFrame:
    """
    Load and merge hourly pollutant and wind data with station coordinates and compute wind components.
    """
    import os
    import pandas as pd

    # Load coordinates
    coord_path = os.path.join(data_dir, station_csv)
    df = pd.read_csv(coord_path, skipinitialspace=True)

    # Load measurements
    target_and_wind_data = load_hourly_measurements(
        data_dir,
        df,
        prefecture_code,
        year,
        month,
        day,
        hour,
        measurement_columns=[target, 'WS(m/s)', 'WD(16Dir)']
    )

    # Filter valid stations
    df = df[df['station_code'].isin(target_and_wind_data)].copy()

    # Map pollutant and wind data
    df[target] = df['station_code'].map(lambda c: target_and_wind_data[c][target])
    df['WS(m/s)'] = df['station_code'].map(lambda c: target_and_wind_data[c]['WS(m/s)'])
    df['WD(16Dir)'] = df['station_code'].map(lambda c: target_and_wind_data[c]['WD(16Dir)'])

    # Compute wind components
    df['U'], df['V'] = compute_wind_uv(df['WS(m/s)'], df['WD(16Dir)'])

    return df

###############################################################################
def load_station_temporal_features(
    data_dir,
    stations_df,
    prefecture_code,
    year,
    month,
    day,
    hour,
    lags=24,
    target_item="Ox(ppm)"
):
    def safe_last(series):
        """Return last element of a Series or np.nan if empty."""
        return series.iloc[-1] if not series.empty else np.nan

    results = []
    datetime_target = datetime(year, month, day, hour)
    base_features = ['NO(ppm)', 'NO2(ppm)', 'U', 'V']
    all_items = [target_item] + base_features

    for _, row in stations_df.iterrows():
        station_code = row['station_code']
        ym_str = f"{year:04d}{month:02d}"
        filename = f"{ym_str}_{prefecture_code}_{station_code}.csv"
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            print(f"[WARN] File not found: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path, encoding='cp932')
            df["datetime"] = pd.to_datetime(
                df["日付"] + " " + df["時"].astype(str).str.zfill(2),
                format='%Y/%m/%d %H',
                errors='coerce'
            )
            df = df.dropna(subset=["datetime"])
            df = df.set_index("datetime").sort_index()

            if datetime_target not in df.index:
                print(f"[WARN] Missing datetime {datetime_target} for station {station_code}")
                continue
            
            # Valid data check
            cols_to_check = ['Ox(ppm)', 'NO(ppm)', 'NO2(ppm)']
            has_only_dashes = True
            
            for col in cols_to_check:
                if col in df.columns:
                    # Controlla se esistono righe valide (non '-')
                    non_dash_values = df[col].astype(str).str.strip() != "-"
                    if non_dash_values.any():
                        has_only_dashes = False
                        break
            
            if has_only_dashes:
                print(f"[WARN] Station {station_code} contains only missing values ('-'). Consider deleting its files.")
                continue

            # Calcolo U/V dal vento (se disponibili)
            if 'WS(m/s)' in df.columns and 'WD(16Dir)' in df.columns:
                df['U'], df['V'] = compute_wind_uv(df['WS(m/s)'], df['WD(16Dir)'])
            else:
                df['U'] = np.nan
                df['V'] = np.nan

            # Conversione colonne in float
            for col in all_items:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna(subset=all_items)
            df = df.loc[:datetime_target]  # until target date

            if df.empty:
                print(f"[WARN] No usable data before {datetime_target} for station {station_code}")
                continue

            station_name = row['station_name']
            feature_row = {
                "station_code": station_code,
                "station_name": station_name,
                "datetime": datetime_target,
            }

            # Coordinates
            feature_row["longitude"] = row["longitude"]
            feature_row["latitude"] = row["latitude"]

            # target item
            feature_row[target_item] = safe_last(df[target_item])
            
            # Add current values of base features (not just derived ones)
            for item in base_features:
                if item in df.columns and not df[item].empty:
                    feature_row[item] = df[item].iloc[-1]
                else:
                    feature_row[item] = np.nan

            for item in all_items:
                # Lag features
                for l in range(1, lags + 1):
                    col = f"{item}_lag{l}"
                    shifted = df[item].shift(l)
                    feature_row[col] = safe_last(shifted)

                # Rolling mean/std
                roll_mean = df[item].shift(1).rolling(3).mean()
                roll_std = df[item].shift(1).rolling(6).std()
                feature_row[f"{item}_roll_mean_3"] = safe_last(roll_mean)
                feature_row[f"{item}_roll_std_6"] = safe_last(roll_std)

            # Differenze sul target
            for d in [1, 2, 3]:
                diff_series = df[target_item].shift(1).diff(d)
                feature_row[f"{target_item}_diff_{d}"] = safe_last(diff_series)

            for item in base_features:
                diff_series = df[item].shift(1).diff(3)
                feature_row[f"{item}_diff_3"] = safe_last(diff_series)

            # Time-based features
            feature_row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            feature_row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            #feature_row["dayofweek"] = datetime_target.weekday()
            #feature_row["is_weekend"] = int(datetime_target.weekday() >= 5)

            results.append(feature_row)

        except Exception as e:
            print(f"[ERROR] Error processing {station_code}: {e}")

    return pd.DataFrame(results)

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
def compute_wind_uv(ws_series, wd_series):
    # Convert wind direction to uppercase strings, remove whitespace
    wd_series_clean = wd_series.astype(str).str.strip().str.upper()

    # Convert wind speed to numeric, coerce invalids to NaN
    ws_numeric = pd.to_numeric(ws_series, errors='coerce')

    # Pre-assign U and V as NaN
    u = pd.Series(np.nan, index=ws_series.index)
    v = pd.Series(np.nan, index=ws_series.index)

    # Identify non-CALM directions
    is_calm = wd_series_clean == "CALM"
    not_calm = ~is_calm

    # Convert to degrees using mapping
    wd_deg = wd_series_clean[not_calm].map(_direction_to_degrees)

    # Drop invalid directions
    valid_deg = wd_deg.notna() & ws_numeric[not_calm].notna()

    angle_rad = np.deg2rad(wd_deg[valid_deg])
    ws_valid = ws_numeric[not_calm][valid_deg]

    # Compute U, V only for valid rows
    u.loc[valid_deg.index[valid_deg]] = ws_valid * np.sin(angle_rad)
    v.loc[valid_deg.index[valid_deg]] = ws_valid * np.cos(angle_rad)

    return u, v

###############################################################################
def create_geojson_from_records(target, records, opacity=0.8):
    features = []

    for record in records:
        station_code = record["station_code"]
        station_name = record["station_name"]
        lat = record["latitude"]
        lon = record["longitude"]
        value = record[target]
        ts = record["datetime"].strftime("%Y-%m-%dT%H:%M:%S")

        gray_scale = int(255 * (1 - min(max(value, 0), 0.1) / 0.1))
        color = f"rgb({gray_scale},{gray_scale},{gray_scale})"

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "time": ts,
                "popup": f"{station_name} ({station_code})<br>{target}: {value:.3f} ppm",
                "icon": "circle",
                "iconstyle": {
                    "fillColor": color,
                    "fillOpacity": opacity,
                    "stroke": True,
                    "radius": 6,
                    "color": "black",
                    "weight": 1
                }
            }
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


###############################################################################
def compute_bounds_from_df(df, margin_ratio=0.10):
    """
    Compute bounds (with margin) for a single-hour DataFrame.
    """
    x = df['longitude'].values
    y = df['latitude'].values

    lon_min, lon_max = x.min(), x.max()
    lat_min, lat_max = y.min(), y.max()

    lon_margin = (lon_max - lon_min) * margin_ratio
    lat_margin = (lat_max - lat_min) * margin_ratio

    lon_min_exp = lon_min - lon_margin
    lon_max_exp = lon_max + lon_margin
    lat_min_exp = lat_min - lat_margin
    lat_max_exp = lat_max + lat_margin

    return (lat_min_exp, lon_min_exp), (lat_max_exp, lon_max_exp)

###############################################################################
def compute_bounds_from_records(records, margin_ratio=0.10):
    """
    Compute geographic bounds with optional margin based on all available records.

    Parameters:
    - records: list of dicts with 'latitude' and 'longitude' keys
    - margin_ratio: fractional margin to expand bounds (default: 10%)

    Returns:
    - bounds: ((lat_min, lon_min), (lat_max, lon_max))
    """
    df = pd.DataFrame(records)
    x = df['longitude'].values
    y = df['latitude'].values

    lon_min, lon_max = x.min(), x.max()
    lat_min, lat_max = y.min(), y.max()

    lon_margin = (lon_max - lon_min) * margin_ratio
    lat_margin = (lat_max - lat_min) * margin_ratio

    lon_min -= lon_margin
    lon_max += lon_margin
    lat_min -= lat_margin
    lat_max += lat_margin

    return (lat_min, lon_min), (lat_max, lon_max)

###############################################################################
def compute_global_measurements_range(target, records, lower_percentile=5, upper_percentile=95):
    all_values = [r[target] for r in records if target in r]
    vmin = np.percentile(all_values, lower_percentile)
    vmax = np.percentile(all_values, upper_percentile)
    return vmin, vmax

###############################################################################
def generate_idw_images_by_hour(
        target,
        records,
        k=7,
        power=1,
        output_dir="idw_frames",
        image_prefix="idw_",
        num_cells=300,
        overwrite=False
):
    os.makedirs(output_dir, exist_ok=True)
    grouped = defaultdict(list)

    for r in records:
        dt = r["datetime"]
        grouped[dt].append(r)

    bounds = compute_bounds_from_records(records)
    vmin, vmax = compute_global_measurements_range(target, records)

    for dt, group in grouped.items():
        df = pd.DataFrame(group)
        filename = dt.strftime(f"{image_prefix}%Y%m%d_%H.png")
        output_file = os.path.join(output_dir, filename)

        if not overwrite and os.path.exists(output_file):
            print(f"⏭️ Skipped existing: {output_file}")
            continue

        # === Generate IDW interpolated grid ===
        grid = generate_idw_grid(
            target,
            df,
            bounds,
            num_cells,
            k,
            power
        )

        # === Generate confidence overlay image ===
        # Color scale limits (percentiles)
        vmin = np.percentile(df[target], 5)
        vmax = np.percentile(df[target], 95)
        generate_confidence_overlay_image(
            df,
            bounds,
            grid,
            vmin,
            vmax,
            output_file,
            cmap_name='Reds'
        )

        print(f"✅ IDW image created: {output_file}")

    return bounds, vmin, vmax

###############################################################################
def get_variogram_parameters(model_name):
    if model_name == 'linear':
        return {'slope': 0.0005, 'nugget': 0.00001}
    else:
        return {'sill': 0.0001, 'range': 0.1, 'nugget': 0.00001}

###############################################################################
def generate_idw_grid(
    target,
    df,
    bounds,
    num_cells=300,
    k=6,
    power=1.2
):
    """
    Compute a 2D grid of interpolated values using Inverse Distance Weighting (IDW).

    Parameters:
        target: column name of the target variable in df
        df: DataFrame with 'longitude', 'latitude', and measurement columns
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        num_cells: number of grid cells along the longitude axis
        k: number of nearest stations to use
        power: IDW power parameter

    Returns:
        grid: interpolated grid (lat_cells x lon_cells)
    """
    # === Extract data ===
    x = df['longitude'].values
    y = df['latitude'].values
    z = df[target].values
    (lat_min, lon_min), (lat_max, lon_max) = bounds

    # === Compute grid dimensions to preserve square cells ===
    lon_cells = num_cells
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    lat_cells = int(lon_cells * lat_range / lon_range)

    grid_lon = np.linspace(lon_min, lon_max, lon_cells)
    grid_lat = np.linspace(lat_min, lat_max, lat_cells)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

    # === Build KD-Tree for fast nearest neighbor search ===
    tree = cKDTree(np.c_[x, y])
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

    distances, indices = tree.query(grid_points, k=k)

    # === Compute IDW interpolation ===
    if k == 1:
        weights = np.ones_like(distances)
        interpolated_values = z[indices]
    else:
        weights = 1 / np.power(distances + 1e-12, power)
        weights /= weights.sum(axis=1, keepdims=True)
        interpolated_values = np.sum(weights * z[indices], axis=1)

    grid = interpolated_values.reshape(grid_y.shape)  # shape: (lat_cells, lon_cells)

    return grid


###############################################################################
def generate_ordinary_kriging_grid(
    target,
    df,
    bounds,
    num_cells=300,
    variogram_model='linear'
):
    """
    Compute a 2D grid of interpolated values using Simple Kriging.

    Parameters:
        df: DataFrame with 'longitude', 'latitude', measuremants
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        num_cells: number of grid cells along the longitude axis
        variogram_model: variogram model name (e.g. 'linear', 'power', 'gaussian')

    Returns:
        grid: interpolated grid (lat_cells x lon_cells)
    """
    # === Extract data ===
    x = df['longitude'].values
    y = df['latitude'].values
    z = df[target].values
    (lat_min, lon_min), (lat_max, lon_max) = bounds

    # === Compute grid dimensions to preserve square cells ===
    lon_cells = num_cells
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    lat_cells = int(lon_cells * lat_range / lon_range)

    grid_lon = np.linspace(lon_min, lon_max, lon_cells)
    grid_lat = np.linspace(lat_min, lat_max, lat_cells)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

    # === Kriging ===
    variogram_parameters = get_variogram_parameters(variogram_model)
    ordinary_kriging = OrdinaryKriging(
        x, y, z,
        variogram_model,
        variogram_parameters,
        verbose=False,
        enable_plotting=False,
    )

    z_interp, _ = ordinary_kriging.execute('grid', grid_lon, grid_lat)
    grid = np.array(z_interp)  # shape: (lat_cells, lon_cells)

    return grid

###############################################################################
def generate_universal_kriging_grid(
    target,
    df,
    bounds,
    drift_terms,
    num_cells=300,
    variogram_model='linear'
):
    """
    Compute a 2D grid of interpolated values using Universal Kriging.

    Parameters:
        target: name of the column to interpolate (e.g., 'Ox(ppm)')
        df: DataFrame with 'longitude', 'latitude', measurements, and external drift variables
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        drift_terms: list of column names to use as external drift (e.g., ['NO(ppm)', 'NO2(ppm)'])
        num_cells: number of grid cells along the longitude axis
        variogram_model: variogram model name (e.g. 'linear', 'power', 'gaussian')

    Returns:
        grid: interpolated grid (lat_cells x lon_cells)
    """
    # === Extract data ===
    x = df['longitude'].values
    y = df['latitude'].values
    z = df[target].values
    external_drift = df[drift_terms].values

    (lat_min, lon_min), (lat_max, lon_max) = bounds

    # === Compute grid dimensions to preserve square cells ===
    lon_cells = num_cells
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    lat_cells = int(lon_cells * lat_range / lon_range)

    grid_lon = np.linspace(lon_min, lon_max, lon_cells)
    grid_lat = np.linspace(lat_min, lat_max, lat_cells)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

    # === Kriging ===
    variogram_parameters = get_variogram_parameters(variogram_model)

    uk = UniversalKriging(
        x,
        y,
        z,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        drift_terms=drift_terms,
        external_drift=external_drift,
        verbose=False,
        enable_plotting=False
    )

    z_interp, _ = uk.execute(
        style='grid',
        xpoints=grid_lon,
        ypoints=grid_lat
    )

    grid = np.array(z_interp)  # shape: (lat_cells, lon_cells)

    return grid

###############################################################################
def normalize_and_colorize_grid(grid, vmin, vmax, cmap_name):
    norm = np.clip((grid - vmin) / (vmax - vmin), 0, 1)
    cmap = plt.get_cmap(cmap_name)
    color_img = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return color_img

###############################################################################
def compute_confidence_mask(df_coords, grid_shape, bounds, max_distance_km):
    from scipy.spatial import cKDTree

    (lat_min, lon_min), (lat_max, lon_max) = bounds
    lat_cells, lon_cells = grid_shape

    grid_x, grid_y = np.meshgrid(
        np.linspace(lon_min, lon_max, lon_cells),
        np.linspace(lat_min, lat_max, lat_cells)
    )
    grid_coords = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    tree = cKDTree(df_coords)
    dists, _ = tree.query(grid_coords, k=1)
    dist_km = dists * 111
    confidence = np.clip(1 - dist_km / max_distance_km, 0, 1).reshape((lat_cells, lon_cells))
    return confidence

###############################################################################
def create_confidence_overlay(confidence, color_img, low=0.45, high=0.55):
    lat_cells, lon_cells, _ = color_img.shape
    gray_alpha = np.zeros((lat_cells, lon_cells))

    gray_alpha[confidence <= low] = 180
    gray_alpha[confidence >= high] = 0
    fade = (high - confidence[(confidence > low) & (confidence < high)]) / (high - low)
    gray_alpha[(confidence > low) & (confidence < high)] = (fade * 180).astype(np.uint8)

    gray_overlay = np.zeros((lat_cells, lon_cells, 4), dtype=np.uint8)
    gray_overlay[..., :3] = 200
    gray_overlay[..., 3] = gray_alpha.astype(np.uint8)

    rgba = np.zeros((lat_cells, lon_cells, 4), dtype=np.uint8)
    rgba[..., :3] = color_img
    rgba[..., 3] = 255

    fg_alpha = gray_overlay[..., 3:4] / 255.0
    rgba[..., :3] = (
        gray_overlay[..., :3] * fg_alpha + rgba[..., :3] * (1 - fg_alpha)
    ).astype(np.uint8)

    return rgba

###############################################################################
def plot_confidence_image(
    rgba,
    df,
    bounds,
    output_file,
    title=None,
    add_wind=True,
    bbox=True,
    station_marker=True
):
    x = df['longitude'].values
    y = df['latitude'].values
    (lat_min, lon_min), (lat_max, lon_max) = bounds

    fig, ax = plt.subplots(figsize=(8, 8 * (lat_max - lat_min) / (lon_max - lon_min)), dpi=200)
    ax.imshow(rgba, extent=(lon_min, lon_max, lat_min, lat_max), origin='lower')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=10)

    if bbox:
        rect = Rectangle(
            (x.min(), y.min()),
            x.max() - x.min(),
            y.max() - y.min(),
            linewidth=1.5,
            edgecolor='black',
            linestyle='--',
            facecolor='none'
        )
        ax.add_patch(rect)

    if station_marker:
        ax.scatter(x, y, c='black', edgecolor='white', s=60, label='Stations')

    if add_wind and 'U' in df.columns and 'V' in df.columns:
        ax.quiver(
            df['longitude'], df['latitude'],
            df['U'], df['V'],
            angles='xy', scale_units='xy', scale=100.0,
            color='blue', width=0.003, label='Wind'
        )

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Saved confidence overlay to {output_file}")

###############################################################################
def generate_confidence_overlay_image(
    df,
    bounds,
    grid,
    vmin,
    vmax,
    output_file,
    cmap_name='viridis',
    max_distance_km=15,
    title=None,
    add_wind=True,
    bbox=True,
    station_marker=True
):
    x = df['longitude'].values
    y = df['latitude'].values
    df_coords = np.column_stack((x, y))

    color_img = normalize_and_colorize_grid(grid, vmin, vmax, cmap_name)
    confidence = compute_confidence_mask(df_coords, grid.shape, bounds, max_distance_km)
    rgba = create_confidence_overlay(confidence, color_img)
    plot_confidence_image(rgba, df, bounds, output_file, title, add_wind, bbox, station_marker)

###############################################################################
def idw_grid_vectorized(df, grid_lon, grid_lat, features, p=2):
    """
    Compute IDW interpolation for all non-geographic features in df
    across a full grid, in a vectorized manner.

    Returns:
        dict: { feature_name: 2D array interpolated on the grid }
    """

    # Coordinate matrices: shape (Ny, Nx)
    glon = grid_lon
    glat = grid_lat

    # Station coordinates: shape (N,)
    slon = df["longitude"].values.reshape(1, 1, -1)
    slat = df["latitude"].values.reshape(1, 1, -1)

    # =============================
    # Distance matrix: (Ny, Nx, N)
    # =============================
    dist = np.sqrt((glon[..., None] - slon)**2 +
                   (glat[..., None] - slat)**2)

    # Avoid division by zero
    dist = np.where(dist == 0, 1e-12, dist)

    # Weight matrix
    w = 1 / (dist ** p)

    results = {}

    # =============================
    # IDW for each non-geo feature
    # =============================
    for f in features:
        if f in ("longitude", "latitude"):
            continue

        vals = df[f].values.reshape(1, 1, -1)

        num = np.sum(w * vals, axis=2)
        den = np.sum(w, axis=2)

        results[f] = num / den

    return results
