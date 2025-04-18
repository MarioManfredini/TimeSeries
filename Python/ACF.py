# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utility import WD_COLUMN

###############################################################################
# === Caricamento dati ===
from utility import load_and_prepare_data

data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################

# Imposta il numero massimo di lag
max_lags = 48

# Crea i plot ACF multipli
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20))
axes = axes.flatten()

for i, col in enumerate(items):
    if not col == WD_COLUMN:
        plot_acf(data[col], lags=max_lags, ax=axes[i], title=f"ACF - {col}")
        plot_pacf(data[col], lags=max_lags, ax=axes[i], title=f"ACF PACF - {col}")

# Elimina gli assi vuoti se ci sono
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
