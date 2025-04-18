# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
# === ① Windrose con concentrazione ===
plt.figure(figsize=(8, 8))
ax = WindroseAxes.from_ax()
wind_and_item = data[['WD_degrees', 'WS(m/s)', target_item]].dropna()
ax.bar(wind_and_item['WD_degrees'], wind_and_item[target_item],
       normed=True, opening=0.8, edgecolor='white',
       colors=['#cccccc', '#aaaaaa', '#999999', '#666666', '#333333', '#000000'],
       bins=np.linspace(wind_and_item[target_item].min(), wind_and_item[target_item].max(), 6))
ax.set_legend(title=target_item, decimal_places=3)
ax.set_title('Wind Rose: Concentrazione Ox(ppm)')
plt.show()

# === ② Scatterplot U/V con concentrazione  ===
plt.figure(figsize=(8, 6))
scatter = plt.scatter(data['U'], data['V'], c=data[target_item], cmap='Greys', alpha=0.7)
cbar = plt.colorbar(scatter)
cbar.set_label(target_item)
plt.xlabel('U (m/s)')
plt.ylabel('V (m/s)')
plt.title('Distribuzione di Ox in base alle componenti vento')
plt.grid(True)
plt.show()

# === ③ Heatmap Direzione-Velocità ===
wind_and_item['Dir_bin'] = pd.cut(wind_and_item['WD_degrees'], bins=np.arange(0, 361, 22.5), right=False)
wind_and_item['Speed_bin'] = pd.cut(wind_and_item['WS(m/s)'], bins=np.arange(0, wind_and_item['WS(m/s)'].max() + 1, 1), right=False)

heatmap_data = wind_and_item.pivot_table(index='Dir_bin', columns='Speed_bin',
                                 values=target_item, aggfunc='mean', observed=False)

plt.figure(figsize=(12, 8))
plt.imshow(heatmap_data, aspect='auto', cmap='Greys', origin='lower')
plt.colorbar(label=target_item)
plt.xticks(ticks=np.arange(len(heatmap_data.columns)),
           labels=[f"{x.left:.1f}" for x in heatmap_data.columns], rotation=90)
plt.yticks(ticks=np.arange(len(heatmap_data.index)),
           labels=[f"{x.left:.1f}" for x in heatmap_data.index])
plt.title('Heatmap Direzione-Velocità vs Ox(ppm)')
plt.xlabel('Velocità vento (m/s)')
plt.ylabel('Direzione vento (gradi)')
plt.show()

# === ④ Grafico 3D U/V/Ox ===
wind_and_item = data[['WD_degrees', 'WS(m/s)', 'U', 'V', target_item]].dropna()

fig = plt.figure(figsize=(10, 8))
ax3d = fig.add_subplot(111, projection='3d')

valid_3d = wind_and_item.dropna(subset=['U', 'V', target_item])

p = ax3d.scatter(wind_and_item['U'], wind_and_item['V'], wind_and_item[target_item],
                 c=wind_and_item[target_item], cmap='Greys', alpha=0.7)
ax3d.set_xlabel('U (m/s)')
ax3d.set_ylabel('V (m/s)')
ax3d.set_zlabel(target_item)
fig.colorbar(p, ax=ax3d, label=target_item)
plt.title('Grafico 3D: Vento e {target_item}')
plt.show()

# === ⑤ Ox(ppm) e WS(m/s) nel tempo ===
plt.figure(figsize=(14, 6))

# Linea Ox(ppm)
plt.plot(data.index, data[target_item], label=target_item, color='black', alpha=0.7)

# Linea WS(m/s) su asse secondario
plt.twinx()
plt.plot(data.index, data['WS(m/s)'], label='WS(m/s)', color='grey', linestyle='dashed', alpha=0.5)

plt.title('{target_item} e Velocità del Vento nel Tempo')
plt.xlabel('Data')
plt.ylabel('{target_item} / WS(m/s)')
plt.legend(loc='upper left')
plt.show()


###############################################################################
from sklearn.linear_model import LinearRegression

X = data[items].drop(columns=[target_item, WD_COLUMN])
y = data[items][target_item]

# Divisione temporale
split_index = int(len(X) * 0.7)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Modello senza lag
model_plain = LinearRegression()
model_plain.fit(X_train, y_train)

y_pred_plain = model_plain.predict(X_test)

###############################################################################
from sklearn.metrics import r2_score

r2_plain = r2_score(y_test, y_pred_plain)
print(f"\nR² senza variabili ritardate: {r2_plain:.5f}")

# Imposta dimensione figura
plt.figure(figsize=(14, 6))

# Primo subplot: senza lag
plt.subplot(1, 2, 1)
plt.plot(y_test.values, label='Valori Reali', color='black')
plt.plot(y_pred_plain, label='Predetti (senza lag)', color='red', linestyle='dashed')
plt.title(f'Regressione senza lag\nR²: {r2_plain:.5f}')
plt.xlabel('Campioni')
plt.ylabel(target_item)
plt.legend()
plt.tight_layout()
plt.show()

residuals_plain = y_test.values - y_pred_plain

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(residuals_plain, bins=30, color='red', alpha=0.7, edgecolor='black')
plt.title('Errori Residui - Modello Senza Lag')
plt.xlabel('Errore')
plt.ylabel('Frequenza')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calcola MAE, MSE, RMSE
mae = mean_absolute_error(y_test, y_pred_plain)
mse = mean_squared_error(y_test, y_pred_plain)
rmse = np.sqrt(mse)

# Stampa i risultati
print(f"\nmedia misurazioni: {np.average(data[target_item]):.5f}")
print(f"MAE (Errore Assoluto Medio): {mae:.5f}")
print(f"MSE (Errore Quadratico Medio): {mse:.5f}")
print(f"RMSE (Radice Errore Quadratico Medio): {rmse:.5f}")