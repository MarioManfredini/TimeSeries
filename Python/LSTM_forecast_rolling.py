# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import matplotlib.pyplot as plt
from utility import load_and_prepare_data
import json

# === Parametri ===
MODEL_PATH = "lstm_best_model.pth"
SCALER_PATH = "lstm_scaler.joblib"
PARAMS_PATH = "lstm_best_params.json"

with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    params = json.load(f)
features = params["features"]
target_item = params["target_item"]

# === Caricamento dati forecast ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38201090'  # Cambia qui per forecast su altra stazione
data_all, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

# === Feature derivate ===
df = data_all.copy()
df[f'{target_item}_diff_1'] = df[target_item].diff(1)
df[f'{target_item}_lag1'] = df[target_item].shift(1)
df[f'{target_item}_lag2'] = df[target_item].shift(2)
df[f'{target_item}_roll_mean_3'] = df[target_item].rolling(window=3).mean()
df['station_id'] = int(station_code)

df_selected = df[features].dropna()

# === Caricamento scaler salvato e normalizzazione ===
scaler = load(SCALER_PATH)
scaled = scaler.transform(df_selected)
scaled_df = pd.DataFrame(scaled, columns=df_selected.columns)

X_scaled = scaled_df[features]
y_scaled = scaled_df[target_item]

# === Creazione delle sequenze ===
def create_sequences(X, y, lag):
    Xs, ys = [], []
    for i in range(len(X) - lag):
        Xs.append(X.iloc[i:i+lag].values)
        ys.append(y.iloc[i+lag])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, params["lag"])

X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1)

# === Definizione modello LSTM ===
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

input_size = X_tensor.shape[2]
model = LSTMModel(input_size, params["hidden_size"], params["num_layers"], params["dropout"])
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# === Previsione ===
with torch.no_grad():
    y_pred = model(X_tensor).squeeze().numpy()
    y_true = y_tensor.squeeze().numpy()

# === Inversione normalizzazione solo target ===
target_index = features.index(target_item)
target_scaler = MinMaxScaler()
target_scaler.min_ = np.array([scaler.min_[target_index]])
target_scaler.scale_ = np.array([scaler.scale_[target_index]])

y_pred_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_true_inv = target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

# === Metriche ===
r2 = r2_score(y_true_inv, y_pred_inv)
mae = mean_absolute_error(y_true_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))

print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# === Grafico ===
plt.plot(y_true_inv[:300], label="True (ppm)")
plt.plot(y_pred_inv[:300], label="Predicted (ppm)")
plt.legend()
plt.title(f"Forecast su stazione {station_code}")
plt.show()
