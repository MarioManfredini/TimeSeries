# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from utility import load_and_prepare_data

# ----------------------
# Config
# ----------------------
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = "38205010"
target_item = 'Ox(ppm)'

sequence_length = 48
forecast_horizon = 12
batch_size = 64
hidden_size = 128
num_layers = 2
dropout = 0.2
learning_rate = 0.001
num_epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Dataset
# ----------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, horizon):
        self.X, self.y = [], []
        for i in range(len(data) - seq_len - horizon + 1):
            x_seq = data[i:i+seq_len]
            y_seq = data[i+seq_len:i+seq_len+horizon, 0]  # target è la prima colonna
            self.X.append(x_seq)
            self.y.append(y_seq)
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------------
# GRU Model
# ----------------------
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=12, dropout=0.2):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

# -----------------------
# Training function
# -----------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10):
    best_r2 = -np.inf
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds = model(x_val)
                val_preds.append(preds.cpu().numpy())
                val_targets.append(y_val.cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_r2 = r2_score(val_targets, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        print(f"Validation R2: {val_r2:.4f}, RMSE: {val_rmse:.4f}")

        if val_r2 > best_r2:
            best_r2 = val_r2
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model based on validation R².")

    return model

# ----------------------
# Load Data
# ----------------------
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

features = [target_item]

# Crea feature derivate complete
lagged_items = [target_item, 'NO(ppm)', 'NO2(ppm)', 'U', 'V']
lags = 3
for item in lagged_items:
    for l in range(1, lags + 1):
        df[f'{item}_lag{l}'] = df[item].shift(l)
        features.append(f'{item}_lag{l}')

# Medie mobili (rolling mean)
df[f'{target_item}_roll_mean_3'] = df[target_item].rolling(3).mean()
features.append(f'{target_item}_roll_mean_3')
df['NO(ppm)_roll_mean_3'] = df['NO(ppm)'].rolling(3).mean()
features.append('NO(ppm)_roll_mean_3')
df['NO2(ppm)_roll_mean_3'] = df['NO2(ppm)'].rolling(3).mean()
features.append('NO2(ppm)_roll_mean_3')
df['U_roll_mean_3'] = df['U'].rolling(3).mean()
features.append('U_roll_mean_3')
df['V_roll_mean_3'] = df['V'].rolling(3).mean()
features.append('V_roll_mean_3')

# Deviazioni standard mobili (rolling std)
df[f'{target_item}_roll_std_6'] = df[target_item].rolling(6).std()
features.append(f'{target_item}_roll_std_6')
df['NO(ppm)_roll_std_6'] = df['NO(ppm)'].rolling(6).std()
features.append('NO(ppm)_roll_std_6')
df['NO2(ppm)_roll_std_6'] = df['NO2(ppm)'].rolling(6).std()
features.append('NO2(ppm)_roll_std_6')
df['U_roll_std_6'] = df['U'].rolling(6).std()
features.append('U_roll_std_6')
df['V_roll_std_6'] = df['V'].rolling(6).std()
features.append('V_roll_std_6')

# Differenze temporali
df[f'{target_item}_diff_1'] = df[target_item].diff(1)
features.append(f'{target_item}_diff_1')
df[f'{target_item}_diff_2'] = df[target_item].diff(2)
features.append(f'{target_item}_diff_2')
df[f'{target_item}_diff_3'] = df[target_item].diff(3)
features.append(f'{target_item}_diff_3')
df['NO(ppm)_diff_3'] = df['NO(ppm)'].diff(3)
features.append('NO(ppm)_diff_3')
df['NO2(ppm)_diff_3'] = df['NO2(ppm)'].diff(3)
features.append('NO2(ppm)_diff_3')
df['U_diff_3'] = df['U'].diff(3)
features.append('U_diff_3')
df['V_diff_3'] = df['V'].diff(3)
features.append('V_diff_3')

# Feature temporali
df["hour_sin"] = np.sin(2 * np.pi * df["時"] / 24)
features.append('hour_sin')
df["hour_cos"] = np.cos(2 * np.pi * df["時"] / 24)
features.append('hour_cos')
df["dayofweek"] = df.index.dayofweek
features.append('dayofweek')
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
features.append('is_weekend')

df.dropna(inplace=True)

data = df[features].values

# Normalize
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_data, columns=features, index=df.index)

# Split
n = len(scaled_df)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

train_data = scaled_df.iloc[:train_end].values
val_data = scaled_df.iloc[train_end:val_end].values
test_data = scaled_df.iloc[val_end:].values

train_dataset = TimeSeriesDataset(train_data, sequence_length, forecast_horizon)
val_dataset = TimeSeriesDataset(val_data, sequence_length, forecast_horizon)
test_dataset = TimeSeriesDataset(test_data, sequence_length, forecast_horizon)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ----------------------
# Train
# ----------------------
model = GRUModel(input_size=len(features), hidden_size=hidden_size,
                 output_size=forecast_horizon, num_layers=num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

# ----------------------
# Evaluation
# ----------------------
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

r2 = r2_score(targets, predictions)
rmse = np.sqrt(mean_squared_error(targets, predictions))
print(f"Validation R2: {r2:.4f}, RMSE: {rmse:.4f}")

with torch.no_grad():
    last_seq = torch.tensor(test_data[-sequence_length:], dtype=torch.float32).unsqueeze(0).to(device)
    forecast = model(last_seq).cpu().numpy().flatten()
    forecast_inverse = scaler.inverse_transform(
        np.concatenate([forecast.reshape(-1, 1), np.zeros((forecast_horizon, len(features)-1))], axis=1)
    )[:, 0]  # solo la colonna del target
    print(f"Forecast 12h (inverso): {forecast_inverse}")
