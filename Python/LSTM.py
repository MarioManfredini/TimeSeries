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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utility import load_and_prepare_data

# === Parametri ===
import json

LAG = 24
EPOCHS = 100
BATCH_SIZE = 128
LR = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
PATIENCE = 5

features = ['NOx(ppm)', 'U', 'V', 'TEMP(℃)', 'HUM(％)', 'Ox(ppm)']

PARAMS_PATH = "best_lstm_params.json"
params = {
    "lag": LAG,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "dropout": DROPOUT,
    "patience": PATIENCE,
    "features": features
}
with open("best_lstm_params.json", "w") as f:
    json.dump(params, f, indent=4, ensure_ascii=False)

# === Caricamento dati ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data_all, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

# === Selezione feature ===
data_selected = data_all[features].dropna()

# === Normalizzazione multivariata ===
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_selected)
X_scaled = data_scaled[:, :-1]  # tutte tranne target
y_scaled = data_scaled[:, -1]   # solo target

# === Creazione sequenze ===
def create_sequences(X, y, lag):
    Xs, ys = [], []
    for i in range(len(X) - lag):
        Xs.append(X[i:i+lag])
        ys.append(y[i+lag])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, LAG)
X_tensor = torch.tensor(X_seq, dtype=torch.float32)  # [samples, seq_len, num_features]
y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1)

# === Dataset ===
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# === LSTM Multivariato ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # prendiamo solo l'ultima uscita
        out = self.fc(out)
        return out

input_size = X_tensor.shape[2]
model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === Early Stopping ===
best_val_loss = float('inf')
epochs_no_improve = 0
MODEL_PATH = "best_lstm_model.pth"

for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validazione
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            val_loss += loss_fn(pred, yb).item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss:.6f}")

    # Controllo miglioramento
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_PATH)  # Salvataggio del miglior modello
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping a epoch {epoch+1}. Miglior Val Loss: {best_val_loss:.6f}")
            break

with open(PARAMS_PATH, "w") as f:
    json.dump(params, f, indent=4, ensure_ascii=False)

# Carica il miglior modello trovato
model.load_state_dict(torch.load(MODEL_PATH))

# === Valutazione finale ===
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).squeeze().numpy()
    y_true = y_tensor.squeeze().numpy()

# === Inversione della normalizzazione ===
# Estrai solo la colonna target (Ox) dallo scaler
ox_scaler = MinMaxScaler()
ox_scaler.min_, ox_scaler.scale_ = scaler.min_[-1:], scaler.scale_[-1:]

y_pred_inv = ox_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_true_inv = ox_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

# === Metriche sulla scala originale ===
r2 = r2_score(y_true_inv, y_pred_inv)
mae = mean_absolute_error(y_true_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))

print("R^2 (original scale):", r2)
print("MAE (original scale):", mae)
print("RMSE (original scale):", rmse)

# Salva su file
with open("best_lstm_result.txt", "w", encoding="utf-8") as f:
    f.write(f"R² (original scale): {r2:.6f}\n")
    f.write(f"MAE (original scale): {mae:.6f}\n")
    f.write(f"RMSE (original scale): {rmse:.6f}\n")

# === Grafico ===
plt.plot(y_true_inv[:200], label="True (ppm)")
plt.plot(y_pred_inv[:200], label="Predicted (ppm)")
plt.legend()
plt.title("Confronto tra valori reali e predetti (ppm)")
plt.show()
