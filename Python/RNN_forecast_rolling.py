# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from utility import load_and_prepare_data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# === Parametri ===
LAG = 24
model_path = "rnn_best_model.pth"
target_item = 'Ox(ppm)'

features = [
    'Ox(ppm)',
    'Ox(ppm)_diff_1',
    'Ox(ppm)_lag1',
    'Ox(ppm)_lag2',
    'Ox(ppm)_roll_mean_3',
]

# === Caricamento dati ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data_all, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

# === Feature derivate ===
df = data_all.copy()
df['Ox(ppm)_diff_1'] = df[target_item].diff(1)
df['Ox(ppm)_lag1'] = df[target_item].shift(1)
df['Ox(ppm)_lag2'] = df[target_item].shift(2)
df['Ox(ppm)_roll_mean_3'] = df[target_item].rolling(window=3).mean()

df_selected = df[features].dropna()

# === Normalizzazione ===
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_selected.values.reshape(-1, 1)).flatten()

# === Definizione modello ===
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = SimpleRNN()
model.load_state_dict(torch.load(model_path))
model.eval()

# === Rolling forecast ===
window = scaled[:LAG].tolist()  # inizializzazione
forecast = []

for i in range(LAG, len(scaled)):
    seq_input = torch.tensor(window[-LAG:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        pred = model(seq_input).item()
    forecast.append(pred)
    #window.append(scaled[i])  # append vero valore
    window.append(pred) # Per forecast puro (senza vero valore):

# === Allineamento target
y_true = scaled[LAG:]

# === Inversione scala
y_pred_inv = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
y_true_inv = scaler.inverse_transform(np.array(y_true).reshape(-1, 1)).flatten()

# === Valutazione
print("Forecast Rolling Results:")
print("R²:  ", r2_score(y_true_inv, y_pred_inv))
print("MAE: ", mean_absolute_error(y_true_inv, y_pred_inv))
print("RMSE:", np.sqrt(mean_squared_error(y_true_inv, y_pred_inv)))

# === Plot
plt.figure(figsize=(10, 4))
plt.plot(y_true_inv[:200], label='True (ppm)')
plt.plot(y_pred_inv[:200], label='Forecast (ppm)')
plt.title("Forecast Rolling vs True")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
