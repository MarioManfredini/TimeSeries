# -*- coding: utf-8 -*-
"""
Created: 2025/07/12
Author: Mario
Description: Moran's I global and local (LISA) with PDF report output.
"""

import os
import tempfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from libpysal.weights import DistanceBand
from esda.moran import Moran, Moran_Local
from matplotlib.backends.backend_pdf import PdfPages
from map_utils import load_station_temporal_features

###############################################################################
def save_moran_summary_pdf(target_variable, moran, year, month, day, hour, fig, filename="moran_summary.pdf"):
    """
    Generate a 1-page PDF summarizing Moran's I results, including a matplotlib plot.
    """
    with PdfPages(filename) as pdf:
        page = plt.figure(figsize=(8.27, 11.69))  # A4 size

        # Titolo principale
        page.text(0.5, 0.94, "Spatial Autocorrelation Analysis (Global Moran's I)", fontsize=12, ha='center')

        # Target e dati
        lines = [
            f"Target Variable: {target_variable}",
            f"Date and Hour: {year}-{month:02}-{day:02} {hour:02}:00",
            "",
            f"Moran's I: {moran.I:.4f}",
            f"Expected I: {moran.EI:.4f}",
            f"Z-score: {moran.z_norm:.2f}",
            f"p-value (normal): {moran.p_norm:.4f}",
            f"p-value (Monte Carlo): {moran.p_sim:.4f}",
        ]

        for i, line in enumerate(lines):
            page.text(0.1, 0.88 - i * 0.012, line, fontsize=9, ha='left')

        # Salva il grafico originale come immagine temporanea
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.savefig(tmpfile.name, dpi=200, bbox_inches='tight')
            image_path = tmpfile.name

        # Inserisci il grafico nella pagina PDF
        img = plt.imread(image_path)
        img_ax = page.add_axes([0.05, 0.05, 0.8, 0.70])  # [left, bottom, width, height] in figure coords (0-1)
        img_ax.imshow(img)
        img_ax.axis('off')

        pdf.savefig(page)
        plt.close(page)

        # Rimuovi immagine temporanea
        os.remove(image_path)

    print(f"✅ PDF saved to {filename}")


###############################################################################
# === CONFIGURATION ===
data_dir = '..\\data\\Osaka\\'
prefecture_code = '27'
station_coordinates = 'Stations_Ox.csv'
target = 'Ox(ppm)'
year = 2025
month = 5
day = 12
hour = 12
output_pdf = f"moran_lisa_report_{target}_{year}{month:02}{day:02}_{hour:02}00.pdf"
lisa_image = "lisa_plot.png"

# === Load and prepare data ===
csv_path = os.path.join(data_dir, station_coordinates)
df = pd.read_csv(csv_path, skipinitialspace=True)

df = load_station_temporal_features(
    data_dir, df, prefecture_code,
    year, month, day, hour,
    lags=0, target_item=target
)
print("[DEBUG] Final dataframe shape:", df.shape)

# === Global Moran's I ===
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf_global = gpd.GeoDataFrame(df, geometry=geometry)
w_global = DistanceBand.from_dataframe(gdf_global, threshold=0.45, silence_warnings=True)
w_global.transform = 'r'

moran = Moran(gdf_global[target], w_global, permutations=999)
print("\n=== Global Moran's I ===")
print(f"Moran's I: {moran.I:.4f}")
print(f"Expected I: {moran.EI:.4f}")
print(f"Z-score: {moran.z_norm:.2f}")
print(f"p-value (normal): {moran.p_norm:.4f}")
print(f"p-value (Monte Carlo): {moran.p_sim:.4f}")

# === Local Moran (LISA) ===
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
    crs="EPSG:4326"
)
gdf = gdf.to_crs(epsg=3857)  # metric projection for distance
w_local = DistanceBand.from_dataframe(gdf, threshold=30000, silence_warnings=True)
y = gdf[target].values
moran_loc = Moran_Local(y, w_local)

# Add LISA results
gdf["local_I"] = moran_loc.Is
gdf["p_sim"] = moran_loc.p_sim
gdf["significant"] = moran_loc.p_sim < 0.05
gdf["cluster"] = "Not Significant"

for i in range(len(gdf)):
    if not gdf.loc[i, "significant"]:
        continue
    q = moran_loc.q[i]
    if q == 1:
        gdf.at[i, "cluster"] = "High-High"
    elif q == 2:
        gdf.at[i, "cluster"] = "Low-High"
    elif q == 3:
        gdf.at[i, "cluster"] = "Low-Low"
    elif q == 4:
        gdf.at[i, "cluster"] = "High-Low"

# === Plot LISA cluster map ===
color_map = {
    "High-High": "red",
    "Low-Low": "blue",
    "High-Low": "orange",
    "Low-High": "yellow",
    "Not Significant": "lightgray"
}

fig, ax = plt.subplots(figsize=(10, 8))
for cluster_type, color in color_map.items():
    subset = gdf[gdf["cluster"] == cluster_type]
    subset.plot(ax=ax, color=color, label=cluster_type, markersize=80)

ax.axis('off')
ax.set_title(f"Local Moran's I - LISA Clusters ({target})")
ax.legend()

plt.tight_layout(pad=0)
plt.savefig(lisa_image, dpi=300)
plt.close()
print(f"[INFO] LISA plot saved to: {lisa_image}")

# === Create PDF report ===
save_moran_summary_pdf(target, moran, year, month, day, hour, fig, filename=output_pdf)
