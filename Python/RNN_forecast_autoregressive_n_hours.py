# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utility import load_and_prepare_data
import json

# === Parametri ===
N_FORECAST = 2  # ore da predire
LAG = 24
target_item = 'Ox(ppm)'

# === Carica parametri modello ===
with open("rnn_best_params.json", "r", encoding="utf-8") as f:
    params = json.load(f)
features = params["features"]

# === Caricamento dati ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

# === Crea feature derivate ===
df['Ox(ppm)_diff_1'] = df[target_item].diff(1)
df['Ox(ppm)_lag1'] = df[target_item].shift(1)
df['Ox(ppm)_lag2'] = df[target_item].shift(2)
df['Ox(ppm)_roll_mean_3'] = df[target_item].rolling(window=3).mean()
df = df[features].dropna()

# === Normalizzazione ===
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df.values.reshape(-1, 1)).flatten()

# === Prepara finestra iniziale ===
initial_window = scaled[-LAG:].tolist()

# === Carica modello ===
class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

model = SimpleRNN()
model.load_state_dict(torch.load("rnn_best_model.pth"))
model.eval()

# === Forecast autoregressivo ===
forecast_scaled = []
window = initial_window.copy()

for _ in range(N_FORECAST):
    x_input = torch.tensor(window[-LAG:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        pred = model(x_input).item()
    forecast_scaled.append(pred)
    window.append(pred)

# === Inversione normalizzazione ===
forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

# === Confronto con valori reali ===
true_values = df[target_item].values[-N_FORECAST:]

r2 = r2_score(true_values, forecast)
mae = mean_absolute_error(true_values, forecast)
rmse = np.sqrt(mean_squared_error(true_values, forecast))

print("Forecast Autoregressivo - Confronto con valori reali")
print(f"R²:   {r2}")
print(f"MAE:  {mae}")
print(f"RMSE: {rmse}")

# === Plot confronto ===
plt.figure(figsize=(10, 4))
plt.plot(true_values, label="Valori Reali", marker='o')
plt.plot(forecast, label="Forecast", marker='x')
plt.title(f"Forecast vs Reale - {N_FORECAST} ore")
plt.xlabel("Ora futura")
plt.ylabel("Ox (ppm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
