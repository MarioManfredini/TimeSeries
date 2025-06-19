# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch_optimizer as optim
import matplotlib.pyplot as plt
from utility import get_station_name, load_and_prepare_data
from report import save_report_to_pdf, plot_comparison_actual_predicted, plot_error_summary_page
import time


start_time = time.time()

###############################################################################
# ---------------------- Config
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = "38206050"
station_name = get_station_name(data_dir, station_code)
target_item = 'Ox(ppm)'

sequence_length = 168 # input window time length (how many consecutive time steps, hours, 168 = 7 days)
forecast_horizon = 24 # 24H = 1 day
batch_size = 128 # number of sequences that are processed together during a single optimization pass (a forward/backward pass).
hidden_size = 64
num_layers = 1
dropout = 0.0
learning_rate = 0.003
num_epochs = 100
patience = 10  # early stopping patience

###############################################################################
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###############################################################################
# Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, target_cols, sequence_length, forecast_horizon):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

        X = []
        y = []

        data = df[feature_cols].values  # shape: (N, num_features)
        target = df[target_cols].values  # shape: (N, forecast_horizon)

        for i in range(len(df) - sequence_length - forecast_horizon + 1):
            X.append(data[i:i + sequence_length])
            y.append(target[i + sequence_length])

        self.X_seq = torch.tensor(X, dtype=torch.float32)
        self.y_seq = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.y_seq[idx]

###############################################################################
# GRU Model
class GRUMultiStep(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast_horizon, dropout=0.2):
        super(GRUMultiStep, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, forecast_horizon)  # Output: a vector of forecast horizon dimensions

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        out, _ = self.gru(x)  # out shape: [batch_size, seq_len, hidden_size]
        out = out[:, -1, :]   # Get the output of the last step of the sequence
        out = self.fc(out)    # shape: [batch_size, forecast_horizon]
        return out

###############################################################################
# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs=20, patience=3):
    best_val_loss = float("inf")
    counter = 0
    best_model = None
    best_epoch = 0

    train_losses_per_epoch = []
    val_losses_per_epoch = []

    for epoch in range(n_epochs):
        model.train()
        train_losses = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)  # output shape: [batch_size, forecast_horizon]
            loss = criterion(output, y_batch)  # y_batch shape: [batch_size, forecast_horizon]
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        train_losses_per_epoch.append(avg_train_loss)

        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                output = model(x_batch)
                loss = criterion(output, y_batch)
                val_losses.append(loss.item())

                all_preds.append(output.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        val_losses_per_epoch.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            best_epoch = epoch
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    if best_model:
        model.load_state_dict(best_model)

    # === Plot Loss ===
    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
    ax.plot(train_losses_per_epoch, label="Training Loss", marker='o', color='lightgray')
    ax.plot(val_losses_per_epoch, label="Validation Loss", marker='x', color='gray')
    ax.axvline(best_epoch, color='lightgray', linestyle='--', label=f"Early stopping @ epoch {best_epoch}")
    ax.set_title("Training and Validation Loss", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss (MSE)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray')
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    plt.close(fig)

    return model, fig

###############################################################################
def evaluate_model_for_report(model, val_loader):
    model.eval()
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            y_true_all.append(y_batch.cpu().numpy())
            y_pred_all.append(output.cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)  # [n_samples, forecast_horizon]
    y_pred = np.concatenate(y_pred_all, axis=0)

    mae_list = []
    rmse_list = []
    r2_list = []
    steps = list(range(1, y_true.shape[1] + 1))

    for step in range(y_true.shape[1]):
        mae = mean_absolute_error(y_true[:, step], y_pred[:, step])
        rmse = np.sqrt(mean_squared_error(y_true[:, step], y_pred[:, step]))
        r2 = r2_score(y_true[:, step], y_pred[:, step])
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)

    return y_true, y_pred, mae_list, rmse_list, r2_list, steps

###############################################################################
# Main
if __name__ == "__main__":

    # ---------------------- Data Preprocessing
    data, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

    # === Feature Engineering ===
    lags = 24
    feature_cols = ['NO(ppm)', 'NO2(ppm)', 'NOx(ppm)', 'U', 'V']
    lagged_items = [target_item, 'NO(ppm)', 'NO2(ppm)', 'U', 'V']
    
    lagged_features = []
    for l in range(1, lags):
        lagged = data[lagged_items].shift(l)
        lagged.columns = [f'{col}_lag{l}' for col in lagged.columns]
        lagged_features.append(lagged)
        
    # Concatenate all lagged features at once
    lagged_df = pd.concat(lagged_features, axis=1)
    feature_cols += list(lagged_df.columns)
    
    # Rolling features
    rolling_features = []
    rolling_windows = [3, 6, 12]
    for window in rolling_windows:
        rolling = data[lagged_items].shift(1).rolling(window)
        mean_cols = rolling.mean().add_suffix(f'_roll_mean_{window}')
        std_cols = rolling.std().add_suffix(f'_roll_std_{window}')
        rolling_features.extend([mean_cols, std_cols])
    
    rolling_df = pd.concat(rolling_features, axis=1)
    feature_cols += list(rolling_df.columns)
    
    # Diff features
    diff_features = pd.DataFrame(index=data.index)
    diff_features[f'{target_item}_diff_1'] = data[target_item].shift(1).diff(1)
    diff_features[f'{target_item}_diff_2'] = data[target_item].shift(1).diff(2)
    diff_features[f'{target_item}_diff_3'] = data[target_item].shift(1).diff(3)
    diff_features[f'{target_item}_diff_cumsum_3'] = data[target_item].shift(1).diff().rolling(3).sum()
    
    for item in ['NO(ppm)', 'NO2(ppm)', 'U', 'V']:
        diff_features[f'{item}_diff_3'] = data[item].shift(1).diff(3)
    
    feature_cols += list(diff_features.columns)
    
    # Time features
    time_features = pd.DataFrame(index=data.index)
    time_features["hour_sin"] = np.sin(2 * np.pi * data["時"] / 24)
    time_features["hour_cos"] = np.cos(2 * np.pi * data["時"] / 24)
    time_features["dayofweek"] = data.index.dayofweek
    time_features["is_weekend"] = (time_features["dayofweek"] >= 5).astype(int)
    feature_cols += list(time_features.columns)
    
    # Ratio feature
    ratio_features = pd.DataFrame(index=data.index)
    ratio_features["NO_ratio"] = data["NO(ppm)"] / (data["NO2(ppm)"] + 1e-5)
    feature_cols += ["NO_ratio"]
    
    # Concatenate all features in one go
    data = pd.concat([data, lagged_df, rolling_df, diff_features, time_features, ratio_features], axis=1)
    
    # Targets
    for i in range(forecast_horizon):
        data[f'{target_item}_t+{i+1:02}'] = data[target_item].shift(-i-1)
    
    target_cols = [f'{target_item}_t+{i+1:02}' for i in range(forecast_horizon)]
    
    # Final clean dataset
    df = data.dropna(subset=feature_cols + target_cols).copy()

    # Features normalization
    scaler_X = StandardScaler() 
    df[feature_cols] = scaler_X.fit_transform(df[feature_cols])

    dataset = TimeSeriesDataset(
                                df,
                                feature_cols,
                                target_cols,
                                sequence_length=sequence_length,
                                forecast_horizon=forecast_horizon
                                )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset = torch.utils.data.Subset(dataset, list(range(train_size)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(train_size, train_size + val_size)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = GRUMultiStep(
                            input_size=len(feature_cols),
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            forecast_horizon=forecast_horizon,
                            dropout=dropout
                        ).to(device)
    criterion = nn.MSELoss()
    base_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Lookahead(base_optimizer, k=5, alpha=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            mode='min',
                            patience=3,
                            factor=0.5
                            )

    figures = []
    
    # Training
    model, fig = train_model(
                            model,
                            train_loader,
                            val_loader,
                            criterion,
                            optimizer,
                            scheduler,
                            n_epochs=num_epochs,
                            patience=patience
                            )
    figures.append(fig)
    
    # Evaluation
    y_true, y_pred, mae_list, rmse_list, r2_list, steps = evaluate_model_for_report(model, val_loader)

    # Report
    figs = plot_comparison_actual_predicted(y_true,
                                            y_pred,
                                            target_cols=steps,
                                            steps=steps,
                                            rows_per_page=3)
    figures.extend(figs)

    fig = plot_error_summary_page(mae_list, rmse_list, r2_list, steps)
    figures.append(fig)

    end_time = time.time()
    elapsed_seconds = int(end_time - start_time)
    elapsed_minutes = elapsed_seconds // 60
    elapsed_seconds_remainder = elapsed_seconds % 60
    elapsed_time_str = f"{elapsed_minutes} min {elapsed_seconds_remainder} sec"
    
    model_params = {
        "Prefecture code": prefecture_code,
        "Station code": station_code,
        "Station name": station_name,
        "Target item": target_item,
        "Number of data points in the train set": train_size,
        "Number of data points in the test set": val_size,
        "Forecast horizon (hours)": forecast_horizon,
        "Model": "GRU",
        "Number of epochs": num_epochs,
        "Elapsed time": elapsed_time_str,
        "Number of used features": len(feature_cols),
        "Features": feature_cols,
    }
    
    errors = [
        (steps[i], r2_score(y_true[:, i], y_pred[:, i]), mae_list[i], rmse_list[i])
        for i in range(forecast_horizon)
    ]

    # === Save PDF ===
    save_report_to_pdf(f'GRU_report_{station_name}_{prefecture_code}_{station_code}_{target_item}.pdf',
                       f'{station_name} - オキシダント予測の分析', model_params, errors, figures)
