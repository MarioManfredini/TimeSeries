# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utility import load_and_prepare_data

# ---------------------- Config
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = "38205010"
target_item = 'Ox(ppm)'

sequence_length = 48
forecast_horizon = 3
batch_size = 128
hidden_size = 64
num_layers = 1
dropout = 0.0
learning_rate = 0.01
num_epochs = 300
patience = 7  # early stopping patience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random

SEED = 42  # o qualsiasi numero fisso
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------- Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, horizon):
        self.X, self.y = [], []
        for i in range(len(data) - seq_len - horizon + 1):
            x_seq = data[i:i+seq_len]
            y_seq = data[i+seq_len:i+seq_len+horizon, 0]
            self.X.append(x_seq)
            self.y.append(y_seq)
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------- GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# ---------------------- Loss Function
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights).float()

    def forward(self, pred, target):
        # pred e target hanno shape [batch_size, forecast_horizon]
        loss = ((pred - target) ** 2) * self.weights.to(pred.device)
        return loss.mean()

# ---------------------- Early Stopping Training
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience):
    best_r2 = -np.inf
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds = model(x_val)
                val_preds.append(preds.cpu().numpy())
                val_targets.append(y_val.cpu().numpy())
                loss = criterion(preds, y_val)
                val_loss += loss.item()

        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_r2 = r2_score(val_targets, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, R2: {val_r2:.4f}, RMSE: {val_rmse:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.6f}")
        
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model based on validation R².")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='x')
    best_epoch = np.argmin(val_losses)
    plt.axvline(best_epoch, color='red', linestyle='--', label=f"Early stopping @ {best_epoch}")
    plt.title("Andamento della Loss durante l'addestramento")
    plt.xlabel("Epoca")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model

# ---------------------- Data Preprocessing
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

# usando 'NO(ppm)', 'NO2(ppm)', 'U', 'V' non laggati peggiora
features = [target_item]

# Feature Engineering
lagged_items = [target_item, 'NO(ppm)', 'NO2(ppm)', 'U', 'V']
for item in lagged_items:
    for l in range(1, 4):
        df[f'{item}_lag{l}'] = df[item].shift(l)
        features.append(f'{item}_lag{l}')

for item in lagged_items:
    df[f'{item}_roll_mean_3'] = df[item].rolling(3).mean()
    features.append(f'{item}_roll_mean_3')
    df[f'{item}_roll_std_6'] = df[item].rolling(6).std()
    features.append(f'{item}_roll_std_6')

df[f'{target_item}_diff_1'] = df[target_item].diff(1)
df[f'{target_item}_diff_2'] = df[target_item].diff(2)
df[f'{target_item}_diff_3'] = df[target_item].diff(3)
features += [f'{target_item}_diff_1', f'{target_item}_diff_2', f'{target_item}_diff_3']

for item in ['NO(ppm)', 'NO2(ppm)', 'U', 'V']:
    df[f'{item}_diff_3'] = df[item].diff(3)
    features.append(f'{item}_diff_3')

df["hour_sin"] = np.sin(2 * np.pi * df["時"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["時"] / 24)
df["dayofweek"] = df.index.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
features += ['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend']

df.dropna(inplace=True)

# ---------------------- Normalize separatamente
target_index = features.index(target_item)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_data = df[features].values
y_data = df[[target_item]].values  # target reshape (n, 1)

scaled_X = scaler_X.fit_transform(X_data)
scaled_y = scaler_y.fit_transform(y_data)

# Sovrascrivo solo la colonna target in scaled_X
scaled_data = scaled_X.copy()
scaled_data[:, target_index] = scaled_y.flatten()

# ---------------------- Split
n = len(scaled_data)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_data = scaled_data[:train_end]
val_data = scaled_data[train_end:val_end]
test_data = scaled_data[val_end:]

train_dataset = TimeSeriesDataset(train_data, sequence_length, forecast_horizon)
val_dataset = TimeSeriesDataset(val_data, sequence_length, forecast_horizon)
test_dataset = TimeSeriesDataset(test_data, sequence_length, forecast_horizon)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ---------------------- Training
model = GRUModel(input_size=len(features), hidden_size=hidden_size,
                 output_size=forecast_horizon, num_layers=num_layers,
                 dropout=dropout).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',        # oppure 'max' se vuoi usarlo su R² (ma 'min' su loss è più stabile)
    factor=0.5,        # dimezza il learning rate
    patience=3,        # aspetta n epoche senza miglioramenti
    min_lr=1e-5        # evita che diventi troppo piccolo
)

model = train_model(model, train_loader, val_loader, criterion, optimizer,
                    num_epochs=num_epochs, patience=patience)

# ---------------------- Evaluation
model.eval()
predictions, targets = [], []
with torch.no_grad():
    for X, y in val_loader:
        X = X.to(device)
        out = model(X)
        predictions.append(out.cpu().numpy())
        targets.append(y.numpy())

predictions = np.concatenate(predictions)
targets = np.concatenate(targets)

# Denormalizza prima del calcolo delle metriche
predictions_inverse = scaler_y.inverse_transform(predictions)
targets_inverse = scaler_y.inverse_transform(targets)

r2 = r2_score(targets_inverse, predictions_inverse)
rmse = np.sqrt(mean_squared_error(targets_inverse, predictions_inverse))
print(f"Validation R2: {r2:.4f}, RMSE: {rmse:.4f}")

# ---------------------- Forecasting
with torch.no_grad():
    last_seq = torch.tensor(test_data[-sequence_length:], dtype=torch.float32).unsqueeze(0).to(device)
    forecast = model(last_seq).cpu().numpy().flatten()
    forecast_inverse = scaler_y.inverse_transform(forecast.reshape(-1, 1)).flatten()
    print(f"Forecast (inverso): {forecast_inverse}")

forecast_preds, forecast_targets = [], []

with torch.no_grad():
    for x_val, y_val in val_loader:
        x_val, y_val = x_val.to(device), y_val.to(device)
        pred = model(x_val)
        forecast_preds.append(pred.cpu().numpy())
        forecast_targets.append(y_val.cpu().numpy())

forecast_preds = np.concatenate(forecast_preds, axis=0)
forecast_targets = np.concatenate(forecast_targets, axis=0)

# Calcolo dell'R² per ciascun passo dell'orizzonte di previsione
r2_per_step = []
for i in range(forecast_preds.shape[1]):
    r2 = r2_score(forecast_targets[:, i], forecast_preds[:, i])
    r2_per_step.append(r2)
    print(f"R² per T+{i+1}: {r2:.4f}")
# Etichette per l'asse x
steps = [f"T+{i+1}" for i in range(forecast_preds.shape[1])]
# Creazione del grafico
plt.figure(figsize=(8, 5))
plt.plot(steps, r2_per_step, marker='o', linestyle='-')
plt.title("R² per ciascun passo dell'orizzonte di previsione")
plt.xlabel("Passo di previsione")
plt.ylabel("R²")
plt.grid(True)
plt.tight_layout()
plt.show()

# Calcola la media su tutte le sequenze previste per ogni passo dell'orizzonte (es: 3 ore)
mean_preds = forecast_preds.mean(axis=0).reshape(-1, 1)
mean_targets = forecast_targets.mean(axis=0).reshape(-1, 1)

# Inverso della normalizzazione
mean_preds_inv = scaler_y.inverse_transform(mean_preds).flatten()
mean_targets_inv = scaler_y.inverse_transform(mean_targets).flatten()

# Grafico
plt.figure(figsize=(8, 5))
plt.plot(mean_targets_inv, label="Valori reali (media)", marker='o')
plt.plot(mean_preds_inv, label="Valori previsti (media)", marker='x')
plt.title("Previsione media")
plt.xlabel("Orizzonte di previsione (ore)")
plt.ylabel("Ox (ppm)")
plt.xticks(ticks=range(len(mean_preds_inv)), labels=[f"T+{i+1}" for i in range(len(mean_preds_inv))])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
