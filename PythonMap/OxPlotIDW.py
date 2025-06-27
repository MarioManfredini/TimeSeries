# -*- coding: utf-8 -*-
"""
Created: 2025/06/23
Author: Mario

Description:
Load Ox(ppm) data from CSV files and visualize interpolated Ox values
over the Ehime region using IDW interpolation (matplotlib grayscale image).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from map_utils import load_latest_ox_values, compute_wind_uv

###############################################################################
# Settings
data_dir = '..\\data\\Osaka\\'
prefecture_code = '27'
station_coordinates = 'Stations_Ox.csv'
csv_path = os.path.join(data_dir, station_coordinates)

# Load station coordinates
df = pd.read_csv(csv_path, skipinitialspace=True)

# Load latest Ox(ppm) values for each station
ox_data = load_latest_ox_values(
    data_dir=data_dir,
    stations_df=df,
    year=2025,
    month=5,
    prefecture_code=prefecture_code
)

# Filter stations with Ox data
# Filter only stations with valid Ox data
df = df[df['station_code'].isin(ox_data.keys())].copy()
df['Ox(ppm)'] = df['station_code'].map(lambda c: ox_data[c]['Ox(ppm)'])
df['WS(m/s)'] = df['station_code'].map(lambda c: ox_data[c]['WS(m/s)'])
df['WD(16Dir)'] = df['station_code'].map(lambda c: ox_data[c]['WD(16Dir)'])

# Compute wind components U and V
df['U'], df['V'] = compute_wind_uv(df['WS(m/s)'], df['WD(16Dir)'])

# Coordinates and Ox values
x = df['longitude'].values
y = df['latitude'].values
z = df['Ox(ppm)'].values

# Create regular grid for interpolation
num_cells = 200
grid_x, grid_y = np.meshgrid(
    np.linspace(x.min(), x.max(), num_cells),
    np.linspace(y.min(), y.max(), num_cells)
)
grid_coords = np.vstack((grid_x.ravel(), grid_y.ravel())).T

# IDW interpolation using k nearest neighbors
tree = cKDTree(np.vstack((x, y)).T)
distances, idx = tree.query(grid_coords, k=5)

weights = 1 / (distances + 1e-12)
weighted_values = np.sum(weights * z[idx], axis=1) / np.sum(weights, axis=1)
grid_z = weighted_values.reshape((num_cells, num_cells))

# Plot interpolated Ox values
plt.figure(figsize=(8, 6))
plt.imshow(
    grid_z,
    extent=(x.min(), x.max(), y.min(), y.max()),
    origin='lower',
    cmap='Greys',
    alpha=0.85
)
plt.colorbar(label='Ox (ppm)', shrink=0.8)

# Overlay station points
plt.scatter(x, y, c='black', edgecolor='white', s=60, label='Stations')

# Add wind vectors
plt.quiver(
    df['longitude'], df['latitude'],
    df['U'], df['V'],
    angles='xy', scale_units='xy', scale=15.0,
    color='skyblue', width=0.003, label='Wind vectors'
)

plt.title("Interpolated Ox (ppm) - IDW")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.tight_layout()
plt.show()
