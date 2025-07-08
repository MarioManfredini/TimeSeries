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
def load_hour_ox_values(data_dir, stations_df, prefecture_code, year, month, day=None, hour=None):
    """
    For each station, load the Ox(ppm), NO(ppm), NO2(ppm), WS(m/s), and WD(16Dir)
    values for a specific date and hour if provided, otherwise take the latest available.

    Parameters:
        data_dir: directory containing the CSVs
        stations_df: DataFrame with a 'station_code' column
        year, month: int, year and month to load
        prefecture_code: str
        day, hour: optional int, specific date/hour to filter (e.g., day=29, hour=23)
    """
    ox_data = {}
    ym_str = f"{year:04d}{month:02d}"

    for _, row in stations_df.iterrows():
        station_code = row['station_code']
        filename = f"{ym_str}_{prefecture_code}_{station_code}.csv"
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            print(f"[WARN] File not found: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path, encoding='cp932')
            df["Ox(ppm)"] = pd.to_numeric(df["Ox(ppm)"], errors='coerce')
            df["NO(ppm)"] = pd.to_numeric(df["NO(ppm)"], errors='coerce')
            df["NO2(ppm)"] = pd.to_numeric(df["NO2(ppm)"], errors='coerce')

            # Drop missing Ox
            df = df[df["Ox(ppm)"].notna()]

            # Datetime parsing
            df["datetime"] = pd.to_datetime(
                df["日付"] + " " + df["時"].astype(str).str.zfill(2),
                format='%Y/%m/%d %H',
                errors='coerce'
            )
            df = df.dropna(subset=["datetime"])

            if day is not None and hour is not None:
                # Filter for specified day/hour
                dt_filter = datetime(year, month, day, hour)
                row_match = df[df["datetime"] == dt_filter]
                if row_match.empty:
                    print(f"[WARN] No data for {dt_filter} at station {station_code}")
                    continue
                target_row = row_match.iloc[0]
            else:
                # Use the latest available row
                df = df.sort_values("datetime")
                target_row = df.iloc[-1]

            # Wind (optional)
            if 'WS(m/s)' in df.columns and 'WD(16Dir)' in df.columns:
                ws = pd.to_numeric(target_row['WS(m/s)'], errors='coerce')
                wd = target_row['WD(16Dir)'] if pd.notna(target_row['WD(16Dir)']) else np.nan
            else:
                ws, wd = np.nan, np.nan

            ox_data[station_code] = {
                'Ox(ppm)': float(target_row["Ox(ppm)"]),
                'NO(ppm)': float(target_row["NO(ppm)"]),
                'NO2(ppm)': float(target_row["NO2(ppm)"]),
                'WS(m/s)': float(ws) if pd.notna(ws) else np.nan,
                'WD(16Dir)': wd
            }

        except Exception as e:
            print(f"[ERROR] Error reading {file_path}: {e}")

    return ox_data


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
    # Pre-assegno U e V a zero dove la direzione è CALM
    u = np.zeros(len(ws_series))
    v = np.zeros(len(ws_series))

    # Maschera delle righe non-CALM
    is_calm = wd_series.str.upper() == "CALM"
    not_calm = ~is_calm

    # Conversione direzione → angolo
    wd_deg = wd_series[not_calm].map(_direction_to_degrees)
    angle_rad = np.deg2rad(wd_deg)

    u[not_calm] = ws_series[not_calm] * np.sin(angle_rad)
    v[not_calm] = ws_series[not_calm] * np.cos(angle_rad)

    return pd.Series(u, index=ws_series.index), pd.Series(v, index=ws_series.index)

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
def generate_idw_images_by_hour(records, k=7, power=1, output_dir="idw_frames", num_cells=800, overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    grouped = defaultdict(list)

    for r in records:
        dt = r["datetime"]
        grouped[dt].append(r)

    bounds = compute_bounds_from_records(records)
    ox_min, ox_max = compute_global_ox_range(records)

    for dt, group in grouped.items():
        df = pd.DataFrame(group)
        filename = dt.strftime("ox_idw_%Y%m%d_%H.png")
        path = os.path.join(output_dir, filename)

        if not overwrite and os.path.exists(path):
            print(f"⏭️ Skipped existing: {path}")
            continue

        generate_ox_confidence_overlay_image(
            df,
            bounds,
            ox_min,
            ox_max,
            k,
            power,
            output_file=path,
            max_distance_km=15,
            cmap_name="Reds"
        )
        print(f"✅ IDW image created: {path}")

    return bounds, ox_min, ox_max


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
def generate_idw_loocv_error_image(df, bounds, k=7, power=1.2,
                                   output_file="loocv_error_idw.png",
                                   num_cells=800, metric='rmse'):
    """
    Interpolate LOOCV errors using IDW and generate a grayscale error image.

    Parameters:
        df: DataFrame with columns ['longitude', 'latitude', 'Ox(ppm)']
        bounds: ((lat_min, lon_min), (lat_max, lon_max))
        k: number of neighbors for IDW
        power: power parameter for IDW
        metric: 'rmse' or 'mae'
    """

    # === Step 1: Compute LOOCV errors ===
    coords = np.vstack((df["longitude"], df["latitude"])).T
    values = df["Ox(ppm)"].values
    errors = []

    for i in range(len(values)):
        x_train = np.delete(coords, i, axis=0)
        z_train = np.delete(values, i)
        point = coords[i].reshape(1, -1)

        tree = cKDTree(x_train)
        distances, idx = tree.query(point, k=k)
        weights = 1 / (distances + 1e-12) ** power
        z_pred = np.sum(weights * z_train[idx]) / np.sum(weights)

        err = values[i] - z_pred
        if metric == 'mae':
            errors.append(abs(err))
        else:  # RMSE by default
            errors.append(err ** 2)

    df["LOOCV_error"] = errors

    # === Step 2: Interpolate errors with IDW ===
    x = df["longitude"].values
    y = df["latitude"].values
    z = df["LOOCV_error"].values

    if metric == 'rmse':
        z = np.sqrt(z)  # convert squared errors to RMSE

    (lat_min, lon_min), (lat_max, lon_max) = bounds
    grid_x, grid_y = np.meshgrid(
        np.linspace(lon_min, lon_max, num_cells),
        np.linspace(lat_min, lat_max, num_cells)
    )
    grid_coords = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    tree = cKDTree(np.vstack((x, y)).T)
    distances, idx = tree.query(grid_coords, k=k)
    weights = 1 / (distances + 1e-12) ** power
    values_interp = np.sum(weights * z[idx], axis=1) / np.sum(weights, axis=1)
    grid_z = values_interp.reshape((num_cells, num_cells))

    # === Step 3: Plot image ===
    aspect_ratio = (lat_max - lat_min) / (lon_max - lon_min)
    fig_width = 8
    fig_height = fig_width * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
    ax.axis('off')
    cmap = 'Greys_r'  # white = high error, black = low error

    im = ax.imshow(
        grid_z,
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin='lower',
        cmap=cmap,
        interpolation='nearest'
    )

    # Station markers
    ax.scatter(x, y, c='red', edgecolor='black', s=60, label='Stations')

    # Colorbar
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label(f"Interpolated LOOCV {metric.upper()}")

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✅ Saved error image to: {output_file}")


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
def generate_ox_confidence_overlay_image(
    df,
    bounds,
    ox_min,
    ox_max,
    k=7,
    power=1.2,
    output_file='ox_idw_confidence.png',
    num_cells=300,
    max_distance_km=20,
    cmap_name='Reds'
):
    x = df['longitude'].values
    y = df['latitude'].values
    z = df['Ox(ppm)'].values
    (lat_min, lon_min), (lat_max, lon_max) = bounds

    # === Create grid with square cells ===
    lon_cells = num_cells
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    lat_cells = int(lon_cells * lat_range / lon_range)

    grid_x, grid_y = np.meshgrid(
        np.linspace(lon_min, lon_max, lon_cells),
        np.linspace(lat_min, lat_max, lat_cells)
    )
    grid_coords = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    # === IDW interpolation ===
    tree = cKDTree(np.vstack((x, y)).T)
    distances, idx = tree.query(grid_coords, k=k)
    weights = 1 / (distances + 1e-12) ** power
    values = np.sum(weights * z[idx], axis=1) / np.sum(weights, axis=1)
    grid_z = values.reshape((lat_cells, lon_cells))  # no flip

    # === Normalize colormap
    norm = np.clip((grid_z - ox_min) / (ox_max - ox_min), 0, 1)
    cmap = plt.get_cmap(cmap_name)
    color_img = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)

    # === Confidence
    dists, _ = tree.query(grid_coords, k=1)
    dist_km = dists * 111
    confidence = np.clip(1 - dist_km / max_distance_km, 0, 1).reshape((lat_cells, lon_cells))

    # === Grayscale alpha mask with soft border
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

    # === Plot
    fig, ax = plt.subplots(figsize=(8, 8 * lat_range / lon_range), dpi=200)
    ax.imshow(rgba, extent=(lon_min, lon_max, lat_min, lat_max), origin='lower')  # no flip
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

    # === Wind vectors
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
    print(f"✅ Saved image with overlays to {output_file}")

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

