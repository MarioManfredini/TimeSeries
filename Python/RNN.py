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
import json

# === Parametri ===
LAG = 24
EPOCHS = 150
BATCH_SIZE = 128
LR = 0.001
HIDDEN_SIZE = 16
PATIENCE = 3

features = [
    'Ox(ppm)',
    'Ox(ppm)_diff_1',
    'Ox(ppm)_lag1',
    'Ox(ppm)_lag2',
    'Ox(ppm)_roll_mean_3',
    ]

target_item = 'Ox(ppm)'

PARAMS_PATH = "rnn_best_params.json"
params = {
    "lag": LAG,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "hidden_size": HIDDEN_SIZE,
    "patience": PATIENCE,
    "features": features
}
with open(PARAMS_PATH, "w") as f:
    json.dump(params, f, indent=4, ensure_ascii=False)

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
#df['TEMP_diff_3'] = df['TEMP(℃)'].diff(3)

# === Selezione delle feature ===
df_selected = df[features].dropna()

# --- Funzione per creare sequenze ---
def create_sequences(series, lag):
    X, y = [], []
    for i in range(len(series) - lag):
        X.append(series[i:i+lag])
        y.append(series[i+lag])
    return np.array(X), np.array(y)

# --- Preparazione dataset ---
scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(df_selected.values.reshape(-1, 1)).flatten()

X, y = create_sequences(scaled_series, LAG)

X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [samples, seq_len, 1]
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # [samples, 1]

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# --- Definizione della RNN ---
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=1):
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

# --- Training con Early Stopping e salvataggio modello ---
best_val_loss = float('inf')
epochs_no_improve = 0
train_losses = []
val_losses = []
best_model_path = "rnn_best_model.pth"

for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0.0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    epoch_train_loss /= len(train_loader)

    # Validazione
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            epoch_val_loss += loss_fn(pred, yb).item()
    epoch_val_loss /= len(val_loader)

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.6f} - Val Loss: {epoch_val_loss:.6f}")

    # Early Stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

# --- Ricarica modello migliore ---
model.load_state_dict(torch.load(best_model_path))
model.eval()

# --- Curva di apprendimento ---
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

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
