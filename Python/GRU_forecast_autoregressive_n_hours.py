# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
from utility import load_and_prepare_data
from GRU_train import GRUNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Parametri
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = "38205010"
target_item = "Ox(ppm)"
lag = 24
n_hours_forecast = 12  # numero di ore da prevedere

# Carica dati
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

# Crea feature derivate complete
lagged_items = [target_item, 'NO(ppm)', 'NO2(ppm)', 'U', 'V']
lags = 3
for item in lagged_items:
    for l in range(1, lags + 1):
        df[f"{item}_lag{l}"] = df[item].shift(l)

df[f'{target_item}_roll_mean_3'] = df[target_item].rolling(3).mean()
df['NO(ppm)_roll_mean_3'] = df['NO(ppm)'].rolling(3).mean()
df['NO2(ppm)_roll_mean_3'] = df['NO2(ppm)'].rolling(3).mean()
df['U_roll_mean_3'] = df['U'].rolling(3).mean()
df['V_roll_mean_3'] = df['V'].rolling(3).mean()

df[f'{target_item}_roll_std_6'] = df[target_item].rolling(6).std()
df['NO(ppm)_roll_std_6'] = df['NO(ppm)'].rolling(6).std()
df['NO2(ppm)_roll_std_6'] = df['NO2(ppm)'].rolling(6).std()
df['U_roll_std_6'] = df['U'].rolling(6).std()
df['V_roll_std_6'] = df['V'].rolling(6).std()

df[f'{target_item}_diff_1'] = df[target_item].diff(1)
df[f'{target_item}_diff_2'] = df[target_item].diff(2)
df[f'{target_item}_diff_3'] = df[target_item].diff(3)
df['NO(ppm)_diff_3'] = df['NO(ppm)'].diff(3)
df['NO2(ppm)_diff_3'] = df['NO2(ppm)'].diff(3)
df['U_diff_3'] = df['U'].diff(3)
df['V_diff_3'] = df['V'].diff(3)

df["hour_sin"] = np.sin(2 * np.pi * df["時"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["時"] / 24)
df["dayofweek"] = df.index.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

df.dropna(inplace=True)

features = [
    f"{target_item}",
    f'{target_item}_lag1',
    f'{target_item}_lag2',
    f'{target_item}_lag3',
    'NO(ppm)_lag1',
    'NO(ppm)_lag2',
    'NO(ppm)_lag3',
    'NO2(ppm)_lag1',
    'NO2(ppm)_lag2',
    'NO2(ppm)_lag3',
    'U_lag1',
    'U_lag2',
    'U_lag3',
    'V_lag1',
    'V_lag2',
    'V_lag3',
    f'{target_item}_roll_mean_3',
    'NO(ppm)_roll_mean_3',
    'NO2(ppm)_roll_mean_3',
    'U_roll_mean_3',
    'V_roll_mean_3',
    f'{target_item}_roll_std_6',
    'NO(ppm)_roll_std_6',
    'NO2(ppm)_roll_std_6',
    'U_roll_std_6',
    'V_roll_std_6',
    f'{target_item}_diff_1',
    f'{target_item}_diff_2',
    f'{target_item}_diff_3',
    'NO(ppm)_diff_3',
    'NO2(ppm)_diff_3',
    'U_diff_3',
    'V_diff_3',
    "hour_sin",
    "hour_cos",
    "dayofweek",
    "is_weekend",
]

data = df[features].values
X_all = []
for i in range(len(data) - lag):
    X_all.append(data[i:i+lag])
X_all = np.array(X_all)

# Carica scaler
scalers = joblib.load("gru_scaler.save")
scaler_X = scalers["X"]
scaler_y = scalers["y"]
feature_names = scalers["feature_names"]

# Normalizza
X_all_scaled = scaler_X.transform(X_all.reshape(-1, X_all.shape[2])).reshape(X_all.shape)

# Punto di partenza per avere sia X che y reali disponibili
start_index = len(X_all_scaled) - n_hours_forecast - 1
current_sequence = X_all_scaled[start_index:start_index+1].copy()

# Valori reali da confrontare
true_values = df[target_item].values[start_index + lag : start_index + lag + n_hours_forecast]

# Carica modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUNet(input_size=X_all.shape[2]).to(device)
model.load_state_dict(torch.load("gru_weights_final.pth"))
model.eval()

# Previsione autoregressiva
predictions_scaled = []
predictions_original = []

for i in range(n_hours_forecast):
    X_tensor = torch.tensor(current_sequence, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy()

    # Inversa normalizzazione
    pred_original = scaler_y.inverse_transform(pred_scaled)[0, 0]
    predictions_scaled.append(pred_scaled[0, 0])
    predictions_original.append(pred_original)

    # Estrai valori necessari dalla sequenza corrente per calcolare nuove feature derivate
    last_step = current_sequence[0, -1, :]
    second_last_step = current_sequence[0, -2, :]
    third_last_step = current_sequence[0, -3, :]
    
    # Inversa normalizzazione dei valori target precedenti
    ox_lag1 = scaler_y.inverse_transform(last_step[0:1].reshape(1, -1))[0, 0]
    ox_lag2 = scaler_y.inverse_transform(second_last_step[0:1].reshape(1, -1))[0, 0]
    ox_lag3 = scaler_y.inverse_transform(third_last_step[0:1].reshape(1, -1))[0, 0]
    
    # Calcolo nuove feature derivate sul target
    ox_diff_1 = pred_original - ox_lag1
    ox_diff_2 = pred_original - ox_lag2
    ox_diff_3 = pred_original - ox_lag3
    ox_roll_mean_3 = (pred_original + ox_lag1 + ox_lag2) / 3
    ox_roll_std_6 = np.std([
        scaler_y.inverse_transform(current_sequence[0, -6:, 0].reshape(-1, 1)).flatten().tolist()
        + [pred_original]
    ]) if current_sequence.shape[1] >= 6 else 0.0
    
    # Costanti/meteo (NO, NO2, U, V e derivate): mantieni gli ultimi valori noti
    def get_feature(name):
        idx = feature_names.index(name)
        return last_step[idx]
    
    # Crea il nuovo vettore di feature
    next_row = []

    # Recupera l'ultima riga nota (non scalata)
    last_step_scaled = current_sequence[0, -1]
    last_step_unscaled = scaler_X.inverse_transform(last_step_scaled.reshape(1, -1))[0]
    
    for name in feature_names:
        if name == f"{target_item}":
            value = pred_original
    
        elif name == f"{target_item}_diff_1":
            value = pred_original - ox_lag1
    
        elif name == f"{target_item}_diff_2":
            value = pred_original - ox_lag2
    
        elif name == f"{target_item}_diff_3":
            value = pred_original - last_step_unscaled[feature_names.index(f"{target_item}_lag3")]
    
        elif name == f"{target_item}_lag1":
            value = ox_lag1
    
        elif name == f"{target_item}_lag2":
            value = ox_lag2
    
        elif name == f"{target_item}_lag3":
            value = last_step_unscaled[feature_names.index(f"{target_item}_lag2")]
    
        elif name == f"{target_item}_roll_mean_3":
            value = (pred_original + ox_lag1 + ox_lag2) / 3
    
        elif name == f"{target_item}_roll_std_6":
            # fallback: usa std dei 3 valori se non hai accesso a 6
            values = [
                pred_original,
                ox_lag1,
                ox_lag2,
                ox_lag3,
            ]
            value = np.nanstd(values)
    
        elif name == "hour_sin":
            value = 0  # opzionalmente: puoi prevedere l'ora futura
    
        elif name == "hour_cos":
            value = 0
    
        elif name == "dayofweek":
            value = 0
    
        elif name == "is_weekend":
            value = 0
    
        else:
            # fallback: mantieni il valore precedente per ogni altra feature
            idx = feature_names.index(name)
            value = last_step_unscaled[idx]
    
        next_row.append(value)
    
    next_row_scaled = scaler_X.transform(np.array(next_row).reshape(1, -1))[0]

    next_sequence = np.concatenate([current_sequence[0, 1:], next_row_scaled.reshape(1, -1)], axis=0)
    current_sequence = next_sequence.reshape(1, lag, -1)

# Grafico
plt.figure(figsize=(12, 6))
plt.plot(predictions_original, label="Predetti (GRU)", color="orange")
if len(true_values) == n_hours_forecast:
    plt.plot(true_values, label="Reali", color="blue")
plt.xlabel("Ora")
plt.ylabel("Ox(ppm)")
plt.title("Previsione autoregressiva vs Valori reali")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Converti in array numpy
y_true = np.array(true_values)
y_pred = np.array(predictions_original)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"Forecast su {len(y_true)} ore:")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

