# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.pipeline import make_pipeline
from utility import WD_COLUMN

target_item = 'Ox(ppm)'

###############################################################################
# === Caricamento dati ===
from utility import load_and_prepare_data

data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################

# Pipeline polinomiale di grado 2
degree = 2
model = make_pipeline(
    PolynomialFeatures(degree=degree, include_bias=False),
    LinearRegression()
)

X = data[items].drop(columns=[target_item, WD_COLUMN])
y = data[items][target_item]

# Time Series Split
tscv = TimeSeriesSplit(n_splits=7)

# Calcolo della learning curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=tscv,
    scoring='r2',
    shuffle=False
)

# Media e deviazione standard
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

# Grafico
plt.figure(figsize=(10,6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="R² Training")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="R² Test")
plt.title(f"Learning Curve - Regressione Polinomiale (grado {degree})\ncon TimeSeriesSplit")
plt.xlabel("Numero di campioni di training")
plt.ylabel("R²")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# Suddividiamo i dati in training e test
split_point = int(len(X) * 0.7)  # 70% training, 30% test
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# Costruiamo polinomi di grado 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Ora i nomi delle colonne li puoi ottenere così:
feature_names = poly.get_feature_names_out(X.columns)

# Regressione
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predizioni
y_pred = model.predict(X_test_poly)

# Valutazione con nomi corretti
coefficients = pd.Series(model.coef_, index=feature_names)

print("\nIntercetta:", model.intercept_)
print("\nCoefficienti:\n")
print(coefficients.sort_values(key=abs, ascending=False))

r2 = r2_score(y_test, y_pred)
print(f"\nR²: {r2:.5f}")

rmse = mean_squared_error(y_test, y_pred)
print(f"\nRMSE (test set): {rmse:.5f}")

# Grafico di confronto
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Valori reali (Ox)')
plt.ylabel('Valori predetti (Ox)')
plt.title(f'Polynomial Regression (grado 2)\nR² = {r2:.5f}')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Linea ideale
plt.grid(True)
plt.show()

# Visualizza confronto tra predetto e reale
plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label='Valori Reali', alpha=0.7)
plt.plot(y_test.index, y_pred, label='Predetti', alpha=0.7)
plt.xlabel('Data')
plt.ylabel(target_item)
plt.title('Confronto Regressione Polinomiale - Split Temporale')
plt.legend()
plt.grid(True)
plt.show()

# Seleziona i coefficienti con valore assoluto maggiore di una soglia (es. 0.01)
threshold = 0.01
important_coeffs = coefficients[coefficients.abs() > threshold]

# Ordina i coefficienti per valore assoluto decrescente
important_coeffs = important_coeffs.reindex(important_coeffs.abs().sort_values(ascending=True).index)

# Crea il grafico a barre orizzontali
plt.figure(figsize=(10, 8))
important_coeffs.plot(kind='barh')
plt.title('Importanza dei Coefficienti nella Regressione Polinomiale')
plt.xlabel('Valore del Coefficiente')
plt.ylabel('Feature')
plt.grid(True)
plt.tight_layout()
plt.show()