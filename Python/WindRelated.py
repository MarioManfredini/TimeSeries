# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from utility import WD_COLUMN

target_item = 'Ox(ppm)'

###############################################################################
# === Caricamento dati ===
from utility import load_and_prepare_data

data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === ① Windrose con concentrazione Ox(ppm) ===
plt.figure(figsize=(8, 8))
ax = WindroseAxes.from_ax()
valid = data[['WD_degrees', 'WS(m/s)', target_item]].dropna()
ax.bar(valid['WD_degrees'], valid[target_item], normed=True, opening=0.8, edgecolor='white',
       colors=['#cccccc', '#aaaaaa', '#999999', '#666666', '#333333', '#000000'],
       bins=np.linspace(valid[target_item].min(), valid[target_item].max(), 6))
ax.set_legend(title='Ox(ppm)', decimal_places=3)
ax.set_title('Wind Rose: Concentrazione Ox(ppm)')
plt.show()

###############################################################################
# === ② Scatterplot U/V con concentrazione Ox(ppm) ===
plt.figure(figsize=(8, 6))
scatter = plt.scatter(data['U'], data['V'], c=data[target_item], cmap='Greys', alpha=0.7)
cbar = plt.colorbar(scatter)
cbar.set_label('Ox(ppm)')
plt.xlabel('U (m/s)')
plt.ylabel('V (m/s)')
plt.title('Distribuzione di Ox in base alle componenti vento')
plt.grid(True)
plt.show()

###############################################################################
# === ③ Heatmap Direzione-Velocità ===
valid['Dir_bin'] = pd.cut(valid['WD_degrees'], bins=np.arange(0, 361, 22.5), right=False)
valid['Speed_bin'] = pd.cut(valid['WS(m/s)'], bins=np.arange(0, valid['WS(m/s)'].max() + 1, 1), right=False)

heatmap_data = valid.pivot_table(index='Dir_bin', columns='Speed_bin',
                                 values=target_item, aggfunc='mean', observed=False)

plt.figure(figsize=(12, 8))
plt.imshow(heatmap_data, aspect='auto', cmap='Greys', origin='lower')
plt.colorbar(label='Ox(ppm)')
plt.xticks(ticks=np.arange(len(heatmap_data.columns)),
           labels=[f"{x.left:.1f}" for x in heatmap_data.columns], rotation=90)
plt.yticks(ticks=np.arange(len(heatmap_data.index)),
           labels=[f"{x.left:.1f}" for x in heatmap_data.index])
plt.title('Heatmap Direzione-Velocità vs Ox(ppm)')
plt.xlabel('Velocità vento (m/s)')
plt.ylabel('Direzione vento (gradi)')
plt.show()

###############################################################################
# === ④ Grafico 3D U/V/Ox ===
valid = data[['WD_degrees', 'WS(m/s)', 'U', 'V', target_item]].dropna()

fig = plt.figure(figsize=(10, 8))
ax3d = fig.add_subplot(111, projection='3d')

valid_3d = valid.dropna(subset=['U', 'V', target_item])

p = ax3d.scatter(valid['U'], valid['V'], valid[target_item],
                 c=valid[target_item], cmap='Greys', alpha=0.7)
ax3d.set_xlabel('U (m/s)')
ax3d.set_ylabel('V (m/s)')
ax3d.set_zlabel('Ox(ppm)')
fig.colorbar(p, ax=ax3d, label='Ox(ppm)')
plt.title('Grafico 3D: Vento e Ox(ppm)')
plt.show()

###############################################################################
# === ⑤ Ox(ppm) e WS(m/s) nel tempo ===
plt.figure(figsize=(14, 6))

# Linea Ox(ppm)
plt.plot(data.index, data[target_item], label='Ox(ppm)', color='black', alpha=0.7)

# Linea WS(m/s) su asse secondario
plt.twinx()
plt.plot(data.index, data['WS(m/s)'], label='WS(m/s)', color='grey', linestyle='dashed', alpha=0.5)

plt.title('Ox(ppm) e Velocità del Vento nel Tempo')
plt.xlabel('Data')
plt.ylabel('Ox(ppm) / WS(m/s)')
plt.legend(loc='upper left')
plt.show()

###############################################################################
from sklearn.linear_model import LinearRegression

X = data[items].drop(columns=[target_item, WD_COLUMN])
y = data[items][target_item]
model = LinearRegression().fit(X, y)

print(f"\nCoefficiente U: {model.coef_[0]:.5f}")
print(f"Coefficiente V: {model.coef_[1]:.5f}")
print(f"Intercetta: {model.intercept_:.5f}")
print(f"\nR²: {model.score(X, y):.5f}")
