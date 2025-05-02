# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utility import load_and_prepare_data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import random

SEED = 42  # o qualsiasi numero fisso
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parametri
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = "38205010"
target_item = "Ox(ppm)"
lag = 24
test_ratio = 0.2
val_ratio = 0.2
batch_size = 128
epochs = 30
learning_rate = 0.0005

# Dataset PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Modello GRU
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1, dropout=0.0):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.gru(x)
        out = self.bn(h[-1])
        out = self.fc(out)
        return out

# Carica dati
df, valid_items = load_and_prepare_data(data_dir, prefecture_code, station_code)

# Crea feature derivate
for i in range(1, 3):
    df[f"{target_item}_lag{i}"] = df[target_item].shift(i)
df[f"{target_item}_diff_1"] = df[target_item].diff(1)
df[f"{target_item}_roll_mean_3"] = df[target_item].rolling(window=3).mean()
df["hour_sin"] = np.sin(2 * np.pi * df["時"]/24)
df["hour_cos"] = np.cos(2 * np.pi * df["時"]/24)
df["dayofweek"] = df.index.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df.dropna(inplace=True)

# Selezione feature
features = [
    f"{target_item}",
    f"{target_item}_diff_1",
    f"{target_item}_lag1",
    f"{target_item}_lag2",
    f"{target_item}_roll_mean_3",
    "hour_sin",
    "hour_cos",
    "dayofweek",
    "is_weekend",
    "NOx(ppm)",
]
data = df[features].values

# Costruzione sequenze
X_all, y_all = [], []
for i in range(len(data) - lag):
    X_all.append(data[i:i+lag])
    y_all.append(data[i+lag][0])

X_all = np.array(X_all)
y_all = np.array(y_all).reshape(-1, 1)

# Split train/val/test PRIMA della normalizzazione
X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=test_ratio, shuffle=False)
val_size = int(len(X_temp) * val_ratio)
X_train_raw, X_val_raw = X_temp[:-val_size], X_temp[-val_size:]
y_train_raw, y_val_raw = y_temp[:-val_size], y_temp[-val_size:]

# Normalizzazione input
X_train_2d = X_train_raw.reshape(-1, X_all.shape[2])
scaler_X = MinMaxScaler()
scaler_X.fit(X_train_2d)
X_train_scaled = scaler_X.transform(X_train_2d).reshape(X_train_raw.shape)
X_val_scaled = scaler_X.transform(X_val_raw.reshape(-1, X_all.shape[2])).reshape(X_val_raw.shape)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_all.shape[2])).reshape(X_test.shape)

# Normalizzazione target
scaler_y = MinMaxScaler()
scaler_y.fit(y_train_raw)
y_train_scaled = scaler_y.transform(y_train_raw)
y_val_scaled = scaler_y.transform(y_val_raw)
y_test_scaled = scaler_y.transform(y_test)

# Salva scaler
joblib.dump({'X': scaler_X, 'y': scaler_y}, "gru_scaler.save")

# DataLoader
train_loader = DataLoader(TimeSeriesDataset(X_train_scaled, y_train_scaled), batch_size=batch_size, shuffle=False)
val_loader = DataLoader(TimeSeriesDataset(X_val_scaled, y_val_scaled), batch_size=batch_size, shuffle=False)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUNet(input_size=X_all.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_losses, val_losses = [], []
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss_avg = train_loss / len(train_loader)
    train_losses.append(train_loss_avg)

    # Valutazione
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()
    val_loss_avg = val_loss / len(val_loader)
    val_losses.append(val_loss_avg)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}")

    # Early stopping
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        counter = 0
        torch.save(model.state_dict(), "gru_weights_best.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# Grafico curva di apprendimento
plt.figure(figsize=(10, 6))
plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Curva di Apprendimento')
plt.legend()
plt.show()

# Salva modello
torch.save(model.state_dict(), "gru_weights_final.pth")
print("Modello GRU salvato con successo.")
# carica subito e verifica
model2 = GRUNet(input_size=X_all.shape[2]).to(device)
model2.load_state_dict(torch.load("gru_weights_final.pth"))
print("Loaded weights final:", list(model2.parameters())[0][0][:5])

# Valutazione finale
model.eval()
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).cpu().numpy()

y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test_scaled)

# Metriche in scala normalizzata
r2_scaled = r2_score(y_test_scaled, y_pred_scaled)
mae_scaled = mean_absolute_error(y_test_scaled, y_pred_scaled)
rmse_scaled = np.sqrt(mean_squared_error(y_test_scaled, y_pred_scaled))

# Metriche in scala originale
r2 = r2_score(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))

print("=== Normalized Scale ===")
print(f"R²: {r2_scaled:.6f}, MAE: {mae_scaled:.6f}, RMSE: {rmse_scaled:.6f}")
print("=== Original Scale ===")
print(f"R²: {r2:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")

# Salva risultati
with open("gru_best_result.txt", "w", encoding="utf-8") as f:
    f.write("=== Normalized Scale ===\n")
    f.write(f"R²: {r2_scaled:.6f}, MAE: {mae_scaled:.6f}, RMSE: {rmse_scaled:.6f}\n")
    f.write("=== Original Scale ===\n")
    f.write(f"R²: {r2:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}\n")

# Grafico confronto
plt.plot(y_test_original[:300], label="True (ppm)")
plt.plot(y_pred_original[:300], label="Predicted (ppm)")
plt.legend()
plt.title("Confronto tra valori reali e predetti (ppm)")
plt.show()
