# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import matplotlib.pyplot as plt
import json
from utility import load_and_prepare_data

# === Parametri ===
MODEL_PATH = "lstm_best_model.pth"
SCALER_PATH = "lstm_scaler.joblib"
PARAMS_PATH = "lstm_best_params.json"
N_FORECAST = 24

# === Caricamento parametri ===
with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    params = json.load(f)

features = params["features"]
target_item = params["target_item"]
lag = params["lag"]
station_code = "38201090"  # Cambia qui per altre stazioni

# === Caricamento dati ===
data_dir = '..\\data\\Ehime\\'
data_all, _ = load_and_prepare_data(data_dir, params["prefecture_code"], station_code)

# === Costruzione feature derivate ===
df = data_all.copy()
df[f'{target_item}_diff_1'] = df[target_item].diff(1)
df[f'{target_item}_lag1'] = df[target_item].shift(1)
df[f'{target_item}_lag2'] = df[target_item].shift(2)
df[f'{target_item}_roll_mean_3'] = df[target_item].rolling(window=3).mean()
df['station_id'] = int(station_code)

df_selected = df[features].dropna().copy()

# === Caricamento scaler ===
scaler = load(SCALER_PATH)
scaled = scaler.transform(df_selected)
scaled_df = pd.DataFrame(scaled, columns=df_selected.columns)

# === Ultima finestra di input ===
window = scaled_df.iloc[-lag:].copy()

# === Definizione modello ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

input_size = len(features)
model = LSTMModel(input_size, params["hidden_size"], params["num_layers"], params["dropout"])
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# === Forecast autoregressivo ===
forecast_scaled = []
for step in range(N_FORECAST):
    input_seq = torch.tensor(window[-lag:].values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(input_seq).squeeze().item()
    forecast_scaled.append(pred_scaled)

    # Creazione nuovo punto con features aggiornate
    last_row = window.iloc[-1].copy()
    new_row = last_row.copy()
    new_row[target_item] = pred_scaled
    new_row[f'{target_item}_lag1'] = window.iloc[-1][target_item]
    new_row[f'{target_item}_lag2'] = window.iloc[-2][target_item]
    new_row[f'{target_item}_diff_1'] = new_row[target_item] - new_row[f'{target_item}_lag1']
    new_row[f'{target_item}_roll_mean_3'] = np.mean([
        window.iloc[-1][target_item],
        window.iloc[-2][target_item],
        window.iloc[-3][target_item]
    ])
    window = pd.concat([window, pd.DataFrame([new_row])], ignore_index=True)

# === Inversione normalizzazione ===
target_index = features.index(target_item)
target_scaler = MinMaxScaler()
target_scaler.min_ = np.array([scaler.min_[target_index]])
target_scaler.scale_ = np.array([scaler.scale_[target_index]])
forecast_pred = target_scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

# === Valori reali per confronto ===
true_df = df_selected[target_item].iloc[-N_FORECAST:].reset_index(drop=True)
true_values = target_scaler.inverse_transform(true_df.values.reshape(-1, 1)).flatten()

# === Metriche ===
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(true_values, forecast_pred)
mae = mean_absolute_error(true_values, forecast_pred)
rmse = np.sqrt(mean_squared_error(true_values, forecast_pred))

print("\nForecast Autoregressivo - Confronto con valori reali")
print(f"R²:   {r2}")
print(f"MAE:  {mae}")
print(f"RMSE: {rmse}")

# === Grafico confronto ===
plt.figure(figsize=(10, 5))
plt.plot(true_values, label="True (ppm)", marker='o')
plt.plot(forecast_pred, label="Predicted (ppm)", marker='x')
plt.title(f"Forecast Autoregressivo - Stazione {station_code}")
plt.xlabel("Ore")
plt.ylabel("Ox (ppm)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
