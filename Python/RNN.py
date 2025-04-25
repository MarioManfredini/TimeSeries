# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from utility import load_and_prepare_data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# === Caricamento dati ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data_all, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

# === Colonne usate ===
cols = ['SO2(ppm)', 'NO(ppm)', 'NO2(ppm)', 'NMHC(ppmC)', 'CH4(ppmC)',
        'SPM(mg/m3)', 'PM2.5(μg/m3)', 'U', 'V', 'TEMP(℃)', 'HUM(％)', 'Ox(ppm)']
target_item = 'Ox(ppm)'
ox_series = data_all[cols][target_item].dropna()  # Solo la colonna Ox(ppm)

# --- Parametri ---
LAG = 3
EPOCHS = 100
BATCH_SIZE = 128
LR = 0.001

# --- Funzione per creare sequenze ---
def create_sequences(series, lag):
    X, y = [], []
    for i in range(len(series) - lag):
        X.append(series[i:i+lag])
        y.append(series[i+lag])
    return np.array(X), np.array(y)

# --- Preparazione dataset ---
scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(ox_series.values.reshape(-1, 1)).flatten()

X, y = create_sequences(scaled_series, LAG)

X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [samples, seq_len, 1]
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # [samples, 1]

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# --- Definizione della RNN ---
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Prendiamo solo l'ultimo timestep
        out = self.fc(out)
        return out

model = SimpleRNN()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Training ---
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Valutazione
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            val_loss += loss_fn(pred, yb).item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss:.6f}")

# --- Valutazione finale ---
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).squeeze().numpy()
    y_true = y_tensor.squeeze().numpy()

# --- Inversione della normalizzazione ---
# y_pred e y_true hanno shape [n_samples], quindi reshape necessario
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

# --- Valutazione sulle scale originali ---
print("R^2 (original scale):", r2_score(y_true_inv, y_pred_inv))
print("MAE (original scale):", mean_absolute_error(y_true_inv, y_pred_inv))
print("RMSE (original scale):", np.sqrt(mean_squared_error(y_true_inv, y_pred_inv)))

# --- Visualizzazione su scala originale ---
plt.plot(y_true_inv[:200], label="True (ppm)")
plt.plot(y_pred_inv[:200], label="Predicted (ppm)")
plt.legend()
plt.title("Confronto tra valori reali e predetti in scala originale")
plt.show()
