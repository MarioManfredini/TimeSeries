# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utility import load_and_prepare_data
import json

# === Parametri ===
LAG = 24
EPOCHS = 100
BATCH_SIZE = 128
LR = 0.001
HIDDEN_SIZE = 32
NUM_LAYERS = 2
DROPOUT = 0.5
PATIENCE = 3

# === Caricamento dati di più stazioni ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_codes = ['38205010',
                 '38201020',
                 '38201090',
                 '38201530']
dfs = []

# === Selezione delle feature, incluso station_id ===
target_item = 'Ox(ppm)'
features = [
    f'{target_item}',
    f'{target_item}_diff_1',
    f'{target_item}_lag1',
    f'{target_item}_lag2',
    f'{target_item}_roll_mean_3',
    'NO(ppm)_roll_std_6',
    'NO(ppm)_diff_3',
    'NO(ppm)_roll_mean_3',
    'NO2(ppm)_roll_std_6',
    'NO2(ppm)_diff_3',
    'NO2(ppm)_roll_mean_3',
    'station_id',
    ]

PARAMS_PATH = "lstm_best_params.json"
params = {
    "lag": LAG,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "dropout": DROPOUT,
    "patience": PATIENCE,
    "target_item": target_item,
    "features": features,
    "prefecture_code": prefecture_code,
    "station_codes": station_codes,
}
with open(PARAMS_PATH, "w") as f:
    json.dump(params, f, indent=4, ensure_ascii=False)

for code in station_codes:
    df_tmp, _ = load_and_prepare_data(data_dir, prefecture_code, code)
    df_tmp['station_id'] = int(code)
    dfs.append(df_tmp)

df_all = pd.concat(dfs).sort_index()

# === Feature derivate ===
df_all[f'{target_item}_diff_1'] = df_all[f'{target_item}'].diff()
df_all[f'{target_item}_lag1'] = df_all[f'{target_item}'].shift(1)
df_all[f'{target_item}_lag2'] = df_all[f'{target_item}'].shift(2)
df_all[f'{target_item}_roll_mean_3'] = df_all[f'{target_item}'].rolling(window=3).mean()
df_all['NO(ppm)_roll_std_6'] = df_all['NO(ppm)'].rolling(window=6).std()
df_all['NO(ppm)_diff_3'] = df_all['NO(ppm)'].diff(3)
df_all['NO(ppm)_roll_mean_3'] = df_all['NO(ppm)'].rolling(window=3).mean()
df_all['NO2(ppm)_roll_std_6'] = df_all['NO2(ppm)'].rolling(window=6).std()
df_all['NO2(ppm)_diff_3'] = df_all['NO2(ppm)'].diff(3)
df_all['NO2(ppm)_roll_mean_3'] = df_all['NO2(ppm)'].rolling(window=3).mean()
"""
Se si aggiungono le caretteristiche legate a NOx il modello peggiora (forse troppo numerose)
df_all['NOx(ppm)_lag1'] = df_all['NOx(ppm)'].shift(1)
df_all['NOx(ppm)_lag2'] = df_all['NOx(ppm)'].shift(2)
df_all['NOx(ppm)_roll_mean_3'] = df_all['NOx(ppm)'].rolling(window=3).mean()
df_all['NOx(ppm)_roll_std_6'] = df_all['NOx(ppm)'].rolling(window=6).std()
df_all['NOx(ppm)_diff_3'] = df_all['NOx(ppm)'].diff(3)
"""

df_selected = df_all[features].dropna()

# === Normalizzazione ===
scaler_input = MinMaxScaler()
scaled = scaler_input.fit_transform(df_selected)
scaled_df = pd.DataFrame(scaled, columns=df_selected.columns)

from joblib import dump
dump(scaler_input, "lstm_scaler_input.joblib")

X_scaled = scaled_df[features]
y_scaled = scaled_df[target_item]

# === Creazione delle sequenze ===
def create_sequences(X, y, lag):
    Xs, ys = [], []
    for i in range(len(X) - lag):
        Xs.append(X.iloc[i:i+lag].values)
        ys.append(y.iloc[i+lag])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, LAG)

X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1)

# === Dataset e DataLoader ===
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# === Modello LSTM ===
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
model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === Early Stopping ===
MODEL_PATH = "lstm_best_model.pth"
best_val_loss = float('inf')
epochs_no_improve = 0
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            val_loss += loss_fn(pred, yb).item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping a epoch {epoch+1}. Miglior Val Loss: {best_val_loss:.6f}")
            break

# Grafico della curva di apprendimento
plt.figure(figsize=(10, 6))
plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Curva di Apprendimento')
plt.legend()
plt.show()

with open(PARAMS_PATH, "w") as f:
    json.dump(params, f, indent=4, ensure_ascii=False)

# === Valutazione finale ===
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).squeeze().numpy()
    y_true = y_tensor.squeeze().numpy()

# === Inversione normalizzazione solo target ===
target_index = list(df_selected.columns).index(target_item)
scaler_target = MinMaxScaler()
scaler_target.min_, scaler_target.scale_ = scaler_input.min_[target_index:target_index+1], scaler_input.scale_[target_index:target_index+1]
dump(scaler_target, "lstm_scaler_target.joblib")

y_pred_inv = scaler_target.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_true_inv = scaler_target.inverse_transform(y_true.reshape(-1, 1)).flatten()

# === Metriche ===
r2 = r2_score(y_true_inv, y_pred_inv)
mae = mean_absolute_error(y_true_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))

print(f"R² (original scale): {r2:.6f}")
print(f"MAE (original scale): {mae:.6f}")
print(f"RMSE (original scale): {rmse:.6f}")

# Salva su file
with open("lstm_best_result.txt", "w", encoding="utf-8") as f:
    f.write(f"R² (original scale): {r2:.6f}\n")
    f.write(f"MAE (original scale): {mae:.6f}\n")
    f.write(f"RMSE (original scale): {rmse:.6f}\n")

# === Grafico ===
plt.plot(y_true_inv[:300], label="True (ppm)")
plt.plot(y_pred_inv[:300], label="Predicted (ppm)")
plt.legend()
plt.title("Confronto tra valori reali e predetti (ppm)")
plt.show()

