# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from utility import load_and_prepare_data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import scipy.stats as stats
from matplotlib.figure import Figure

def save_report_to_pdf(filename, model_params, errors, figures):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    
    with PdfPages(filename) as pdf:
        # Page 1: Info and parameters
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size in inches
        ax.axis("off")
        
        text = "Model Parameters:\n"
        for k, v in model_params.items():
            text += f"{k}: {v}\n"
        text += "\nMetrics per Forecast Step:\n"
        for err in errors:
            text += f"{err[0]} - R²: {err[1]:.4f}, MAE: {err[2]:.4f}, RMSE: {err[3]:.4f}\n"

        ax.text(0.01, 0.99, text, va='top', ha='left', fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        # Other pages: charts
        for fig in figures:
            if isinstance(fig, Figure):
                pdf.savefig(fig)
                plt.close(fig)
            else:
                print("- Figure ignored because invalid:", fig)

    print(f"✅ Report saved as: {filename}")

def plot_all_feature_importances(models, target_names, top_n=20):
    num_targets = len(models)
    cols = 2
    rows = (num_targets + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69))  # A4
    axes = axes.flatten()

    for i, (model, target) in enumerate(zip(models, target_names)):
        booster = model.booster_
        importance = booster.feature_importance(importance_type='gain')
        feature_names = booster.feature_name()

        sorted_idx = np.argsort(importance)[::-1][:top_n]
        top_features = np.array(feature_names)[sorted_idx]
        top_importance = importance[sorted_idx]

        ax = axes[i]
        ax.barh(range(top_n), top_importance[::-1], tick_label=top_features[::-1])
        ax.set_xlabel("Importance (gain)", fontsize=8)
        ax.set_title(f"{target}", fontsize=9)

        # Riduci font dei tick
        ax.tick_params(axis='both', which='major', labelsize=7)

    # Rimuove eventuali assi vuoti
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig_to_return = fig
    plt.close(fig)
    return fig_to_return

def plot_top_feature_importance_all_models(multi_model, feature_names, target_names, top_n=20):
    importances_df = pd.DataFrame()

    for i, estimator in enumerate(multi_model.estimators_):
        importance = estimator.feature_importances_
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance,
            'Target': target_names[i]
        })
        importances_df = pd.concat([importances_df, df], ignore_index=True)

    top_features = (
        importances_df
        .sort_values(by="Importance", ascending=False)
        .groupby("Target")
        .head(top_n)
    )

    feature_order = (
        top_features.groupby("Feature")["Importance"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    top_features = top_features[top_features["Feature"].isin(feature_order)]

    num_targets = len(top_features["Target"].unique())
    gray_colors = [str(round(0.1 + 0.8 * i / (num_targets - 1), 2)) for i in range(num_targets)]
    color_map = dict(zip(sorted(target_names), gray_colors))

    fig, ax = plt.subplots(figsize=(14, 8))
    for target in sorted(top_features["Target"].unique()):
        subset = top_features[top_features["Target"] == target]
        subset = subset.set_index("Feature").reindex(feature_order).reset_index()
        ax.plot(subset["Feature"], subset["Importance"], marker='o', label=target, color=color_map[target])

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Top 20 Feature Importances per Target")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    ax.legend(title="Target (grayscale)")
    ax.grid(True)
    fig.tight_layout()

    fig_to_return = fig
    plt.close(fig)
    return fig_to_return

def plot_feature_importance_heatmap(multi_model, feature_names, target_names, top_n=20):
    importance_matrix = pd.DataFrame(index=feature_names)

    for i, estimator in enumerate(multi_model.estimators_):
        importance_matrix[target_names[i]] = estimator.feature_importances_

    top_features = (
        importance_matrix
        .mean(axis=1)
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    heatmap_data = importance_matrix.loc[top_features]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="Greys",
        linewidths=0.5,
        cbar_kws={'label': 'Importance'},
        ax=ax
    )
    ax.set_title(f"Top {top_n} Feature Importance Heatmap per Target")
    ax.set_xlabel("Target")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    fig_to_return = fig
    plt.close(fig)
    return fig_to_return

def plot_feature_importance_heatmap_normalized(multi_model, feature_names, target_names, top_n=20):
    importance_matrix = pd.DataFrame(index=feature_names)

    for i, estimator in enumerate(multi_model.estimators_):
        importance_matrix[target_names[i]] = estimator.feature_importances_

    top_features = (
        importance_matrix
        .mean(axis=1)
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    heatmap_data = importance_matrix.loc[top_features]

    scaler = MinMaxScaler()
    heatmap_normalized = pd.DataFrame(
        scaler.fit_transform(heatmap_data.T).T,
        index=heatmap_data.index,
        columns=heatmap_data.columns
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_normalized,
        annot=True,
        fmt=".2f",
        cmap="Greys",
        linewidths=0.5,
        cbar_kws={'label': 'Normalized Importance (0–1)'},
        ax=ax
    )
    ax.set_title(f"Top {top_n} Feature Importance Heatmap (Normalized by Feature)")
    ax.set_xlabel("Target")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    fig_to_return = fig
    plt.close(fig)
    return fig_to_return

def plot_residuals_grid(y_true, y_pred, target_cols, targets_per_page=4):
    pages = []
    for start in range(0, len(target_cols), targets_per_page):
        end = start + targets_per_page
        sub_cols = target_cols[start:end]

        fig, axes = plt.subplots(len(sub_cols), 3, figsize=(11.69, 8.27))  # A4
        if len(sub_cols) == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, col in enumerate(sub_cols):
            residuals = y_true[:, start + i] - y_pred[:, start + i]

            # Histogram
            sns.histplot(residuals, kde=True, bins=30, ax=axes[i, 0])
            axes[i, 0].set_title(f"{col} - Residual Histogram")
            axes[i, 0].set_xlabel("Residuals")
            axes[i, 0].set_ylabel("Count")

            # Residuals vs Predicted
            axes[i, 1].scatter(y_pred[:, start + i], residuals, alpha=0.3)
            axes[i, 1].axhline(0, color="red", linestyle="--")
            axes[i, 1].set_xlabel("Predicted")
            axes[i, 1].set_ylabel("Residuals")
            axes[i, 1].set_title(f"{col} - Residuals vs Predicted")

            # Q-Q Plot
            stats.probplot(residuals, dist="norm", plot=axes[i, 2])
            axes[i, 2].set_title(f"{col} - Q-Q Plot")

        fig.tight_layout()
        plt.close(fig)
        pages.append(fig)
    
    return pages



###############################################################################
# === Parameters ===
data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
target_item = 'Ox(ppm)'

forecast_horizon = 8  # n-step forecast

figures = []

###############################################################################
# === Load Data ===
df, _ = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
# === Feature Engineering ===
features = [target_item, 'NO(ppm)', 'NO2(ppm)', 'U', 'V']
lagged_items = [target_item, 'NO(ppm)', 'NO2(ppm)', 'U', 'V']

lag_features = {}
for item in lagged_items:
    for l in range(1, 24):
        col_name = f'{item}_lag{l}'
        lag_features[col_name] = df[item].shift(l)
        features.append(col_name)

df = pd.concat([df, pd.DataFrame(lag_features, index=df.index)], axis=1)

rolling_features = {}
for item in lagged_items:
    col_mean = f'{item}_roll_mean_3'
    col_std = f'{item}_roll_std_6'
    rolling_features[col_mean] = df[item].rolling(3).mean()
    rolling_features[col_std] = df[item].rolling(6).std()
    features += [col_mean, col_std]

df = pd.concat([df, pd.DataFrame(rolling_features, index=df.index)], axis=1)

diff_features = {}
diff_features[f'{target_item}_diff_1'] = df[target_item].diff(1)
diff_features[f'{target_item}_diff_2'] = df[target_item].diff(2)
diff_features[f'{target_item}_diff_3'] = df[target_item].diff(3)
features += list(diff_features.keys())

for item in ['NO(ppm)', 'NO2(ppm)', 'U', 'V']:
    col_name = f'{item}_diff_3'
    diff_features[col_name] = df[item].diff(3)
    features.append(col_name)

df = pd.concat([df, pd.DataFrame(diff_features, index=df.index)], axis=1)

df["hour_sin"] = np.sin(2 * np.pi * df["時"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["時"] / 24)
df["dayofweek"] = df.index.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
features += ['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend']

# Create multi-output target
for i in range(forecast_horizon):
    df[f'{target_item}_t+{i+1}'] = df[target_item].shift(-i-1)

target_cols = [f'{target_item}_t+{i+1}' for i in range(forecast_horizon)]

data_model = df.dropna(subset=features + target_cols).copy()

###############################################################################
# === Normalization ===
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_array = scaler_X.fit_transform(data_model[features])
y_array = scaler_y.fit_transform(data_model[target_cols])

X = pd.DataFrame(X_array, columns=features)
y = pd.DataFrame(y_array, columns=target_cols)

###############################################################################
# === Train/Test Split ===
split_index = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

###############################################################################
# === Model Training ===
base_model = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    n_estimators=400,
    learning_rate=0.04,
    max_depth=-1,
    random_state=42, # this number is used to seed the C++ code
)

model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

fig = plot_all_feature_importances(model.estimators_, target_cols)
figures.append(fig)

###############################################################################
# === Forecast ===
X_test_df = pd.DataFrame(X_test, columns=features)
y_pred_scaled = model.predict(X_test_df)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

###############################################################################
# === Evaluation and Residuals ===
plt.figure(figsize=(12, 5))
t_index = data_model.index[split_index:split_index+len(y_pred)]

mae_list = []
rmse_list = []

print("Error per forecast step (t+1, t+2, ...):")
for i, col in enumerate(target_cols):
    r2_scores = r2_score(y_true[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    mae_list.append(mae)
    rmse_list.append(rmse)
    print(f"{col}: R²: {r2_scores:.4f} MAE = {mae:.4f}, RMSE = {rmse:.4f}")
figs = plot_residuals_grid(y_true, y_pred, target_cols, targets_per_page=4)
figures.extend(figs)

steps = ['t+1', 't+2', 't+3', 't+4', 't+5', 't+6', 't+7', 't+8']

fig, ax = plt.subplots(4, 2, figsize=(16, 10))
for i, axi in enumerate(ax.flatten()):
    axi.plot(y_true[:300, i], label='Reale', color='black')
    axi.plot(y_pred[:300, i], label='Predetto', color='red', linestyle='--')
    axi.set_title(f'Ox(ppm) - Step {steps[i]}')
    axi.legend()
fig.suptitle('Comparison between actual and predicted values')
fig.tight_layout()
figures.append(fig)
plt.close(fig)

fig = plot_top_feature_importance_all_models(model, X_train.columns, target_cols, top_n=10)
figures.append(fig)

fig = plot_feature_importance_heatmap(model, X_train.columns, target_cols, top_n=10)
figures.append(fig)

fig = plot_feature_importance_heatmap_normalized(model, X_train.columns, target_cols, top_n=10)
figures.append(fig)

# Plot MAE and RMSE
x = np.arange(len(steps))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width / 2, mae_list, width, label='MAE', color='skyblue')
ax.bar(x + width / 2, rmse_list, width, label='RMSE', color='salmon')
ax.set_xticks(x)
ax.set_xticklabels(steps)
ax.set_title('MAE and RMSE for each forecast step')
ax.legend()
fig.tight_layout()
figures.append(fig)
plt.close(fig)

model_params = {
    "data_dir": data_dir,
    "prefecture_code": prefecture_code,
    "station_code": station_code,
    "target_item": target_item,
    "forecast_horizon": forecast_horizon,
    "features": features,
    "objective": base_model.get_params()['objective'],
    "boosting_type": base_model.get_params()['boosting_type'],
    "n_estimators": base_model.get_params()['n_estimators'],
    "learning_rate": base_model.get_params()['learning_rate'],
    "max_depth": base_model.get_params()['max_depth'],
}

errors = [
    (target_cols[i], r2_score(y_true[:, i], y_pred[:, i]), mae_list[i], rmse_list[i])
    for i in range(forecast_horizon)
]

# === Save PDF ===
save_report_to_pdf("LightGBM_report.pdf", model_params, errors, figures)
