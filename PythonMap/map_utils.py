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
from folium import PolyLine
from folium.plugins import PolyLineTextPath
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from geopy.distance import geodesic
from pykrige.ok import OrdinaryKriging
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
    ox_data = load_hourly_measurements(
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
    df = df[df['station_code'].isin(ox_data)].copy()

    # Map pollutant and wind data
    df[target] = df['station_code'].map(lambda c: ox_data[c][target])
    df['WS(m/s)'] = df['station_code'].map(lambda c: ox_data[c]['WS(m/s)'])
    df['WD(16Dir)'] = df['station_code'].map(lambda c: ox_data[c]['WD(16Dir)'])

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

            feature_row = {
                "station_code": station_code,
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
            feature_row["dayofweek"] = datetime_target.weekday()
            feature_row["is_weekend"] = int(datetime_target.weekday() >= 5)

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
def generate_confidence_overlay_image(
    df,
    bounds,
    target,
    vmin,
    vmax,
    output_file,
    k=7,
    power=1.2,
    num_cells=300,
    max_distance_km=15,
    cmap_name='viridis',
    title=None,
    add_wind=True,
    bbox=True,
    station_marker=True
):
    """
    Generate an interpolated spatial image with confidence overlay.

    Parameters:
        df: DataFrame with at least 'longitude', 'latitude', and value_column
        bounds: ((lat_min, lon_min), (lat_max, lon_max)) geographic bounds
        value_column: str, name of the measurement column (e.g., 'Ox(ppm)')
        vmin, vmax: float, min/max values for normalization
        output_file: str, path of the output PNG file
        k: int, number of neighbors for IDW
        power: float, IDW power parameter
        num_cells: int, number of horizontal grid cells
        max_distance_km: float, for confidence shading
        cmap_name: str, colormap
        title: str or None, optional plot title
        add_wind: bool, whether to draw wind vectors if columns exist
        bbox: bool, whether to draw dashed rectangle bounding stations
        station_marker: bool, whether to draw station markers
    """
    x = df['longitude'].values
    y = df['latitude'].values
    z = df[target].values
    (lat_min, lon_min), (lat_max, lon_max) = bounds

    # === Grid computation
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    lon_cells = num_cells
    lat_cells = int(lon_cells * lat_range / lon_range)

    grid_x, grid_y = np.meshgrid(
        np.linspace(lon_min, lon_max, lon_cells),
        np.linspace(lat_min, lat_max, lat_cells)
    )
    grid_coords = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    # === IDW interpolation
    tree = cKDTree(np.vstack((x, y)).T)
    distances, idx = tree.query(grid_coords, k=k)
    weights = 1 / (distances + 1e-12) ** power
    values = np.sum(weights * z[idx], axis=1) / np.sum(weights, axis=1)
    grid_z = values.reshape((lat_cells, lon_cells))

    # === Normalize colors
    norm = np.clip((grid_z - vmin) / (vmax - vmin), 0, 1)
    cmap = plt.get_cmap(cmap_name)
    color_img = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)

    # === Confidence overlay
    dists, _ = tree.query(grid_coords, k=1)
    dist_km = dists * 111
    confidence = np.clip(1 - dist_km / max_distance_km, 0, 1).reshape((lat_cells, lon_cells))

    # === Alpha fade
    low, high = 0.45, 0.55
    gray_alpha = np.zeros_like(confidence)
    gray_alpha[confidence <= low] = 180
    gray_alpha[confidence >= high] = 0
    fade = (high - confidence[(confidence > low) & (confidence < high)]) / (high - low)
    gray_alpha[(confidence > low) & (confidence < high)] = (fade * 180).astype(np.uint8)

    # === Blend gray overlay
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

    # === Plotting
    fig, ax = plt.subplots(figsize=(8, 8 * lat_range / lon_range), dpi=200)
    ax.imshow(rgba, extent=(lon_min, lon_max, lat_min, lat_max), origin='lower')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=10)

    # === Bounding box
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

    # === Station markers
    if station_marker:
        ax.scatter(x, y, c='black', edgecolor='white', s=60, label='Stations')

    # === Wind vectors
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
    print(f"✅ Saved overlay image to {output_file}")

###############################################################################
def create_geojson_from_records(records, opacity=0.8):
    features = []

    for record in records:
        station_code = record["station_code"]
        station_name = record["station_name"]
        lat = record["latitude"]
        lon = record["longitude"]
        ox = record["Ox(ppm)"]
        ts = record["datetime"].strftime("%Y-%m-%dT%H:%M:%S")

        # Scala del grigio basata su Ox
        gray_scale = int(255 * (1 - min(max(ox, 0), 0.1) / 0.1))
        color = f"rgb({gray_scale},{gray_scale},{gray_scale})"

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "time": ts,
                "popup": f"{station_name} ({station_code})<br>Ox(ppm): {ox:.3f} ppm",
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
def compute_global_ox_range(records, lower_percentile=5, upper_percentile=95):
    all_ox_values = [r['Ox(ppm)'] for r in records if 'Ox(ppm)' in r]
    ox_min = np.percentile(all_ox_values, lower_percentile)
    ox_max = np.percentile(all_ox_values, upper_percentile)
    return ox_min, ox_max

###############################################################################
def generate_idw_image(df, bounds, ox_min, ox_max, k=7, power=1, output_file='ox_idw.png', num_cells=800):
    """
    Generate an IDW interpolated grayscale image over specified bounds.

    Parameters:
        df: DataFrame with columns ['longitude', 'latitude', 'Ox(ppm)', 'U', 'V']
        bounds: tuple ((lat_min, lon_min), (lat_max, lon_max))
        output_file: path to save the resulting PNG
        num_cells: resolution of the interpolation grid
        ox_min, ox_max: fixed min and max values for grayscale (required)
    """
    if ox_min is None or ox_max is None:
        raise ValueError("ox_min and ox_max must be provided to ensure consistent grayscale scale.")

    x = df['longitude'].values
    y = df['latitude'].values
    z = df['Ox(ppm)'].values

    (lat_min, lon_min), (lat_max, lon_max) = bounds

    # Create interpolation grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(lon_min, lon_max, num_cells),
        np.linspace(lat_min, lat_max, num_cells)
    )
    grid_coords = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    # Interpolate using IDW
    tree = cKDTree(np.vstack((x, y)).T)
    distances, idx = tree.query(grid_coords, k=k)
    weights = 1 / (distances + 1e-12) ** power
    values = np.sum(weights * z[idx], axis=1) / np.sum(weights, axis=1)
    grid_z = values.reshape((num_cells, num_cells))

    # Plot
    aspect_ratio = (lat_max - lat_min) / (lon_max - lon_min)
    width = 8
    height = width * aspect_ratio

    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    ax.axis('off')
    ax.imshow(
        grid_z,
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin='lower',
        vmin=ox_min,
        vmax=ox_max,
        cmap='Greys',
        alpha=0.7,
        aspect='auto'
    )

    # Bounding box for original station area
    orig_lon_min, orig_lon_max = x.min(), x.max()
    orig_lat_min, orig_lat_max = y.min(), y.max()

    rect = Rectangle(
        (orig_lon_min, orig_lat_min),
        orig_lon_max - orig_lon_min,
        orig_lat_max - orig_lat_min,
        linewidth=1.5,
        edgecolor='black',
        linestyle='--',
        facecolor='none'
    )
    ax.add_patch(rect)

    # Overlay station markers
    ax.scatter(x, y, c='black', edgecolor='white', s=60, label='Stations')

    # Wind vectors
    ax.quiver(
        df['longitude'], df['latitude'],
        df['U'], df['V'],
        angles='xy', scale_units='xy', scale=100.0,
        color='skyblue', width=0.003, label='Wind vectors'
    )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_file, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

###############################################################################
def generate_idw_images_by_hour(target, records, k=7, power=1, output_dir="idw_frames", num_cells=800, overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    grouped = defaultdict(list)

    for r in records:
        dt = r["datetime"]
        grouped[dt].append(r)

    bounds = compute_bounds_from_records(records)
    vmin, vmax = compute_global_ox_range(records)

    for dt, group in grouped.items():
        df = pd.DataFrame(group)
        filename = dt.strftime("ox_idw_%Y%m%d_%H.png")
        path = os.path.join(output_dir, filename)

        if not overwrite and os.path.exists(path):
            print(f"⏭️ Skipped existing: {path}")
            continue

        generate_confidence_overlay_image(
            df,
            bounds,
            target,
            vmin,
            vmax,
            output_file=filename,
            cmap_name='Reds',
        )
        print(f"✅ IDW image created: {path}")

    return bounds, vmin, vmax


###############################################################################
def draw_arrow_wind_vectors(map_obj, df, scale=0.02, color='red'):
    """
    Draw arrows on a folium map using PolyLineTextPath to represent wind vectors.
    """
    for _, row in df.iterrows():
        if pd.notna(row['U']) and pd.notna(row['V']):
            start = [row['latitude'], row['longitude']]
            end = [
                row['latitude'] + row['V'] * scale,
                row['longitude'] + row['U'] * scale
            ]
            line = PolyLine([start, end], color=color, weight=1)
            line.add_to(map_obj)
            # Add arrow along the line
            PolyLineTextPath(
                line,
                '▶',  # '➤' or try '▶', '➔'
                repeat=True,
                offset=3,
                attributes={'fill': color, 'font-weight': 'bold', 'font-size': '8'}
            ).add_to(map_obj)

###############################################################################
def generate_distance_mask(df, bounds, num_cells=(800, 800), max_distance_km=10.0):
    """
    Generate a confidence mask based on distance from nearest station.
    The mask is vertically flipped to match image orientation.

    Parameters:
        df: DataFrame with station coordinates (columns 'longitude', 'latitude')
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        num_cells: tuple (height, width) for the output mask resolution
        max_distance_km: distance in kilometers beyond which confidence is 0

    Returns:
        2D NumPy array (height, width) with values in [0,1], vertically aligned to image
    """
    # === Grid dimensions ===
    h, w = num_cells

    (lat_min, lon_min), (lat_max, lon_max) = bounds
    grid_x, grid_y = np.meshgrid(
        np.linspace(lon_min, lon_max, w),
        np.linspace(lat_min, lat_max, h)
    )
    grid_coords = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    # Station coordinates
    station_coords = np.vstack((df['longitude'].values, df['latitude'].values)).T
    tree = cKDTree(station_coords)
    distances, _ = tree.query(grid_coords, k=1)

    # Approx. conversion from degree to km (lat-based)
    sample_point = (lat_min, lon_min)
    one_deg_north = (lat_min + 1.0, lon_min)
    km_per_deg = geodesic(sample_point, one_deg_north).km
    distances_km = distances * km_per_deg

    # Normalize to confidence in [0,1]
    norm = Normalize(vmin=0, vmax=max_distance_km, clip=True)
    confidence = 1 - norm(distances_km)
    confidence_image = confidence.reshape((h, w))

    # Flip vertically to match image coordinate system
    return np.flipud(confidence_image)

###############################################################################
def get_variogram_parameters(model_name):
    if model_name == 'linear':
        return {'slope': 0.0005, 'nugget': 0.00001}
    else:
        return {'sill': 0.0001, 'range': 0.1, 'nugget': 0.00001}

###############################################################################
def generate_simple_kriging_grid(
    df,
    bounds,
    num_cells=300,
    variogram_model='linear'
):
    """
    Compute a 2D grid of interpolated Ox(ppm) values using Simple Kriging.

    Parameters:
        df: DataFrame with 'longitude', 'latitude', 'Ox(ppm)'
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        num_cells: number of grid cells along the longitude axis
        variogram_model: variogram model name (e.g. 'linear', 'power', 'gaussian')

    Returns:
        ox_grid: interpolated grid (lat_cells x lon_cells)
        (lat_cells, lon_cells): dimensions of the grid
        grid_x, grid_y: meshgrid arrays (lon and lat coordinates)
    """
    # === Extract data ===
    x = df['longitude'].values
    y = df['latitude'].values
    z = df['Ox(ppm)'].values
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
    params = get_variogram_parameters(variogram_model)
    OK = OrdinaryKriging(
        x, y, z,
        variogram_model=variogram_model,
        variogram_parameters=params,
        verbose=False,
        enable_plotting=False,
    )

    z_interp, _ = OK.execute('grid', grid_lon, grid_lat)
    ox_grid = np.array(z_interp)  # shape: (lat_cells, lon_cells)

    return ox_grid, (lat_cells, lon_cells), grid_x, grid_y

###############################################################################
def generate_ox_kriging_confidence_overlay_image(
    df,
    bounds,
    ox_grid,
    ox_min,
    ox_max,
    output_file='ox_kriging_confidence.png',
    max_distance_km=20,
    cmap_name='Reds'
):
    """
    Visualizes a Kriging result with a semi-transparent gray overlay in low-confidence areas
    based on distance from measurement stations.

    Parameters:
        df: DataFrame with 'longitude', 'latitude', 'Ox(ppm)', and optional 'U', 'V' columns
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        ox_grid: 2D NumPy array with interpolated Ox(ppm) from Kriging (no flip)
        ox_min, ox_max: color scale min/max
        output_file: path to output PNG
        max_distance_km: max distance considered reliable
        cmap_name: colormap for Ox
    """
    x = df['longitude'].values
    y = df['latitude'].values
    (lat_min, lon_min), (lat_max, lon_max) = bounds

    lat_cells, lon_cells = ox_grid.shape

    # === Colormap ===
    norm = np.clip((ox_grid - ox_min) / (ox_max - ox_min), 0, 1)
    cmap = plt.get_cmap(cmap_name)
    color_img = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)

    # === Confidence from distance ===
    from scipy.spatial import cKDTree

    grid_x, grid_y = np.meshgrid(
        np.linspace(lon_min, lon_max, lon_cells),
        np.linspace(lat_min, lat_max, lat_cells)
    )
    grid_coords = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    tree = cKDTree(np.vstack((x, y)).T)
    dists, _ = tree.query(grid_coords, k=1)
    dist_km = dists * 111  # ~111 km per degree
    confidence = np.clip(1 - dist_km / max_distance_km, 0, 1).reshape((lat_cells, lon_cells))

    # === Confidence to gray alpha
    low, high = 0.45, 0.55
    gray_alpha = np.zeros_like(confidence)
    gray_alpha[confidence <= low] = 180
    gray_alpha[confidence >= high] = 0
    fade = (high - confidence[(confidence > low) & (confidence < high)]) / (high - low)
    gray_alpha[(confidence > low) & (confidence < high)] = (fade * 180).astype(np.uint8)

    # === Gray overlay blending
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

    # === Plot ===
    fig, ax = plt.subplots(figsize=(8, 8 * (lat_max - lat_min) / (lon_max - lon_min)), dpi=200)
    ax.imshow(rgba, extent=(lon_min, lon_max, lat_min, lat_max), origin='lower')
    ax.axis('off')

    # === Bounding box
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

    # === Stations
    ax.scatter(x, y, c='black', edgecolor='white', s=60, label='Stations')

    # === Wind
    if 'U' in df.columns and 'V' in df.columns:
        ax.quiver(
            df['longitude'], df['latitude'],
            df['U'], df['V'],
            angles='xy', scale_units='xy', scale=100.0,
            color='blue', width=0.003, label='Wind'
        )

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Saved Kriging confidence overlay image to {output_file}")

