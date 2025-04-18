# -*- coding: utf-8 -*-
"""
Created 2025/04/12

@author: Mario
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

target_item = 'Ox(ppm)'

###############################################################################
# === Caricamento dati ===
from utility import load_and_prepare_data

data_dir = '..\\data\\Ehime\\'
prefecture_code = '38'
station_code = '38205010'
data, items = load_and_prepare_data(data_dir, prefecture_code, station_code)

###############################################################################
plt.rcParams.update({
    'font.family': 'Meiryo',
    'figure.figsize': (10, 5),
    'lines.linewidth': 1,
    'lines.markersize': 3,
    'font.size': 9,
    'legend.fontsize': 8,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'savefig.dpi': 150,
    'figure.dpi': 100
})

###############################################################################
# Plot U vs V scatter
plt.figure(figsize=(8, 8))
plt.scatter(data['U'], data['V'], alpha=0.3, s=10, c='blue')
plt.axhline(0, color='gray', linewidth=0.8)
plt.axvline(0, color='gray', linewidth=0.8)
plt.title('Distribuzione del vento (U-V)')
plt.xlabel('Componente U (m/s)')
plt.ylabel('Componente V (m/s)')
plt.grid(True)
plt.show()

###############################################################################
# Perform seasonal decomposition
###############################################################################
result = seasonal_decompose(data[target_item], model='additive', period=168)
result.plot()
plt.show()

###############################################################################
# ADF test
###############################################################################
# Function to perform ADF test
# Test di Dickey-Fuller aumentato (ADF): Controlla se un trend è presente o meno.
def adf_test(series, title=""):
    result = adfuller(series, autolag='AIC')
    print(f"\nADF Test: {title}")
    print(f"Test Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value}")
    if result[1] < 0.05:
        print("Result: The series is likely stationary.")
    else:
        print("Result: The series is likely non-stationary.")

# Apply ADF test to the original data
adf_test(data[target_item], title="Original Data")

# Apply ADF test to the detrended data (Original - Trend component)
detrended_series = data[target_item] - result.trend
detrended_series.dropna(inplace=True)  # Remove NaN values due to differencing
adf_test(detrended_series, title="Detrended Data")

###############################################################################
# Differenziazione
###############################################################################
# Differenziazione semplice
diff_1 = data[target_item].diff().dropna()
# Test di Dickey-Fuller sulla serie differenziata
adf_test(diff_1, title="First Differencing")

# Differenziazione semplice
diff_2 = diff_1.diff().dropna()
# Test di Dickey-Fuller sulla serie differenziata
adf_test(diff_2, title="Second Differencing")

# Differenziazione stagionale (lag 168 = 1 settimana)
diff_seasonal = data[target_item].diff(periods=168).dropna()
# Test di Dickey-Fuller sulla serie differenziata per stagioni
adf_test(diff_seasonal, title="Seasonal Differencing")

# Plot per confrontare i metodi
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
axes[0].plot(detrended_series, label="Detrended (Trend Removed)", color='blue')
axes[0].set_title("Detrended Series")

axes[1].plot(diff_1, label="First Differencing", color='orange')
axes[1].set_title("First Differencing")

axes[2].plot(diff_2, label="Second Differencing", color='green')
axes[2].set_title("Second Differencing")

axes[3].plot(diff_seasonal, label="Seasonal Differencing", color='black')
axes[3].set_title("Seasonal Differencing")

plt.tight_layout()
plt.show()

###############################################################################
# ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function)
###############################################################################
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_acf(data[target_item], ax=axes[0], title='ACF - Original Data')
plot_pacf(data[target_item], ax=axes[1], title='PACF - Original Data')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_acf(detrended_series, ax=axes[0], title='ACF - Detrended Data')
plot_pacf(detrended_series, ax=axes[1], title='PACF - Detrended Data')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_acf(diff_1, ax=axes[0], title='ACF - First Differencing')
plot_pacf(diff_1, ax=axes[1], title='PACF - First Differencing')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_acf(diff_2, ax=axes[0], title='ACF - Second Differencing')
plot_pacf(diff_2, ax=axes[1], title='PACF - Second Differencing')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_acf(diff_seasonal, ax=axes[0], title='ACF - Seasonal Differencing')
plot_pacf(diff_seasonal, ax=axes[1], title='PACF - Seasonal Differencing')
plt.show()

###############################################################################
"""
def find_d(series, max_d=2):
    adf_result = adfuller(series.dropna())  # Verifica direttamente la serie originale
    if adf_result[1] < 0.05:
        return 0  # La serie è già stazionaria
    
    for d in range(1, max_d + 1):
        diff_series = series.diff(d).dropna()
        adf_result = adfuller(diff_series)
        if adf_result[1] < 0.05:
            return d
    return max_d

d = find_d(data[target_item])
print(f"Optimal differencing (d) found: {d}")

# Plot ACF and PACF for estimating p and q
if d > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(data[target_item].diff(d).dropna(), ax=axes[0], lags=40)
    axes[0].set_title(f"ACF - Differenced d={d}")
    plot_pacf(data[target_item].diff(d).dropna(), ax=axes[1], lags=40)
    axes[1].set_title(f"PACF - Differenced d={d}")
    plt.show()
else:
    print("La serie è già stazionaria, ACF e PACF verranno calcolati direttamente sulla serie originale.")
"""
###############################################################################

#data[target_item] = data[target_item].asfreq('h')

#data[target_item] = np.log(data[target_item])

"""
from pmdarima import auto_arima
stepwise_model = auto_arima(data[target_item], 
                             start_p=0, max_p=6,
                             start_d=0, max_d=4,
                             start_q=0, max_q=5,
                             seasonal=False,
                             trace=True,
                             error_action='ignore',  
                             suppress_warnings=True,
                             stepwise=True)
print(stepwise_model.summary())
"""


# User input for p and q
"""
#p = int(input("Inserisci il valore stimato di p: "))
#q = int(input("Inserisci il valore stimato di q: "))
p = 3
d = 0
q = 1

# Fit ARIMA model
model = ARIMA(data[target_item], order=(p, d, q))
results = model.fit()
print(results.summary())

residuals = results.resid
"""

###############################################################################
# Valutazione dei modelli GARCH
"""
def compare_garch_models(series, p_range=(1, 3), q_range=(1, 3), distributions=["normal", "gaussian", "t", "studentst", "skewt", "ged", "generalized error"]):
    results = []

    for p in range(p_range[0], p_range[1]+1):
        for q in range(q_range[0], q_range[1]+1):
            for dist in distributions:
                try:
                    model = arch_model(series, vol='GARCH', p=p, q=q, dist=dist, rescale=True)
                    fitted = model.fit(disp="off")
                    results.append({
                        "p": p,
                        "q": q,
                        "distribution": dist,
                        "AIC": fitted.aic,
                        "BIC": fitted.bic,
                        "LogLik": fitted.loglikelihood
                    })
                except Exception as e:
                    print(f"Errore con p={p}, q={q}, dist={dist}: {e}")

    return pd.DataFrame(results).sort_values("LogLik")

# Supponiamo `residuals` sia la tua serie residua
best_models_df = compare_garch_models(residuals, p_range=(1, 3), q_range=(1, 3))
print("\nConfronto modelli GARCH:")
print(best_models_df.sort_values(["AIC"]))
"""

def compare_garch_models(
    series, 
    p_range=(1, 3), 
    q_range=(1, 3), 
    distributions=["normal", "gaussian", "studentst", "skewt", "generalized error"],
    lags_for_tests=[10]
):
    results = []

    for p in range(p_range[0], p_range[1]+1):
        for q in range(q_range[0], q_range[1]+1):
            for dist in distributions:
                print(f"Testando p={p}, q={q}, dist={dist}")
                try:
                    model = arch_model(
                        series,
                        mean='ARX',
                        lags=(5, 1),  # modifica se vuoi usare altro
                        vol='GARCH',
                        p=p, q=q,
                        dist=dist,
                        rescale=True
                    )
                    fitted = model.fit(disp="off")
                    
                    # α + β
                    params = fitted.params
                    alpha = params.filter(like='alpha').sum()
                    beta = params.filter(like='beta').sum()
                    alpha_beta = alpha + beta
                    
                    std_resid = fitted.std_resid.dropna()

                    # Ljung-Box
                    lb_test = acorr_ljungbox(std_resid, lags=lags_for_tests, return_df=True)
                    lb_pval = lb_test['lb_pvalue'].median()

                    # ARCH test
                    arch_pval = het_arch(std_resid, nlags=lags_for_tests[-1])[1]

                    results.append({
                        "p": p,
                        "q": q,
                        "distribution": dist,
                        "AIC": fitted.aic,
                        "BIC": fitted.bic,
                        "LogLik": fitted.loglikelihood,
                        "Alpha+Beta": alpha_beta,
                        "Ljung-Box pval": lb_pval,
                        "ARCH pval": arch_pval
                    })

                except Exception as e:
                    print(f"❌ Errore con p={p}, q={q}, dist={dist}: {e}")

    df_results = pd.DataFrame(results)
    return df_results.sort_values(by="AIC")


###############################################################################

def report_garch_models(results_df, top_n=10):
    # Se il DataFrame è vuoto
    if results_df.empty:
        print("⚠️ Nessun modello trovato.")
        return

    # Prendi i top N per AIC
    top_models = results_df.sort_values(by="AIC").head(top_n).copy()

    # Aggiungi colonne di valutazione
    top_models["A+B < 1"] = top_models["Alpha+Beta"] < 1
    top_models["Ljung-Box ok"] = top_models["Ljung-Box pval"] > 0.05
    top_models["ARCH ok"] = top_models["ARCH pval"] > 0.05

    # Rappresentazione con simboli ✓ / ✗
    for col in ["A+B < 1", "Ljung-Box ok", "ARCH ok"]:
        top_models[col] = top_models[col].apply(lambda x: "✓" if x else "✗")

    # Seleziona e rinomina per chiarezza
    display_cols = [
        "p", "q", "distribution", "AIC", "BIC", "LogLik",
        "Alpha+Beta", "Ljung-Box pval", "ARCH pval",
        "A+B < 1", "Ljung-Box ok", "ARCH ok"
    ]

    print(f"\n🔍 Report dei migliori {top_n} modelli GARCH per AIC:\n")
    print(top_models[display_cols].to_string(index=False))

###############################################################################

# Dopo aver generato il DataFrame con la tua funzione:
best_models_df = compare_garch_models(data[target_item], p_range=(1, 3), q_range=(1, 3))

# Mostra solo modelli "validi"
"""
best_models_df = best_models_df[
    (best_models_df["ARCH pval"] > 0.05) &
    (best_models_df["Alpha+Beta"] < 1)
]
"""

report_garch_models(best_models_df, top_n=10)

"""
results = compare_garch_models(data[target_item])

# Mostra solo modelli "validi"
valid_models = results[
    (results["Ljung-Box pval"] > 0.05) &
    (results["ARCH pval"] > 0.05) &
    (results["Alpha+Beta"] < 1)
]

# Ordina i validi per AIC crescente
valid_models = valid_models.sort_values(by="AIC")

print("\nModelli validi (AIC ordinato):")
print(valid_models)
"""



"""

# Fit GARCH(1,1) sui residui dell’ARIMA
garch_model = arch_model(
    data[target_item],  # qui usi direttamente la serie, non i residui!
    mean='ARX',              # oppure 'AR', 'ARMA', ecc.
    lags=(3, 1),             # AR o ARMA: lags=(p, q)
    vol='GARCH',
    p=1, q=1,
    dist='normal',              # o 'normal', 't', 'skewt', ecc.
    rescale=True
)
garch_results = garch_model.fit(disp="off")
# Riassunto del modello GARCH
print("\nGARCH:")
print(garch_results.summary())

std_resid = garch_results.std_resid.dropna()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(std_resid, ax=ax[0], lags=40)
ax[0].set_title('ACF dei residui standardizzati')
plot_pacf(std_resid, ax=ax[1], lags=40)
ax[1].set_title('PACF dei residui standardizzati')
plt.tight_layout()
plt.show()

plot_acf(std_resid**2, lags=40)
plt.title("ACF dei residui standardizzati al quadrato")
plt.show()

import seaborn as sns
import scipy.stats as stats

# Istogramma
sns.histplot(std_resid, kde=True)
plt.title("Distribuzione dei residui standardizzati")
plt.show()

# QQ-plot
stats.probplot(std_resid, dist="norm", plot=plt)
plt.title("QQ-plot dei residui standardizzati (vs Normale)")
plt.show()

lb_test = acorr_ljungbox(std_resid, lags=[10, 20, 30], return_df=True)
print("\nTest di Ljung-Box sui residui standardizzati:")
print(lb_test)

arch_test = het_arch(std_resid)
print("\nTest ARCH sui residui standardizzati:")
print(f"LM stat: {arch_test[0]:.4f}, p-value: {arch_test[1]:.4f}")

"""

"""
print("\nResiduals description:")
print(residuals.describe())  # Controlla min, max, NaN

# Test for normality of residuals
stat, p_value = shapiro(residuals)
print(f"\nShapiro-Wilk test dei residui: p-value={p_value}")

result = anderson(residuals)
print("\nAnderson Residuals:")
print("Statistic:", result.statistic)
print("Critical values:", result.critical_values)
print("Significance levels:", result.significance_level)

# ==== Analisi visiva dei residui ====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(residuals)
axes[0].set_title("Residui del modello ARIMA")
axes[1].hist(residuals, bins=20)
axes[1].set_title("Distribuzione dei residui")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(residuals, ax=axes[0])
plot_pacf(residuals, ax=axes[1])
plt.show()

# ==== Boxplot per individuare outlier ====
plt.figure(figsize=(6, 4))
plt.boxplot(residuals, vert=False)
plt.title("Boxplot dei residui")
plt.show()

# ==== Metodo Z-score (|z| > 3) ====
z_scores = zscore(residuals)
outliers_z = residuals[abs(z_scores) > 3]
print(f"\nOutlier rilevati con Z-score (|z| > 3): {len(outliers_z)}")

# ==== Metodo IQR ====
Q1 = residuals.quantile(0.25)
Q3 = residuals.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = residuals[(residuals < lower_bound) | (residuals > upper_bound)]
print(f"Outlier rilevati con IQR: {len(outliers_iqr)}")

# ==== Visualizzazione con soglie ±3σ ====
mean = residuals.mean()
std = residuals.std()
threshold = 3 * std

plt.figure(figsize=(12, 4))
plt.plot(residuals, label='Residui')
plt.axhline(mean + threshold, color='red', linestyle='--', label='+3σ')
plt.axhline(mean - threshold, color='red', linestyle='--', label='-3σ')
plt.title("Residui con soglie ±3σ")
plt.legend()
plt.show()

from statsmodels.stats.diagnostic import het_arch

# 1. Grafico dei residui in funzione del tempo per valutare la varianza visivamente
plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.title("Controllo visivo: varianza dei residui nel tempo")
plt.xlabel("Tempo")
plt.ylabel("Residui")
plt.grid(True)
plt.show()

# 2. Test di ARCH per eteroschedasticità
arch_test = het_arch(residuals)
print("\nTest di ARCH per eteroschedasticità:")
print(f"Statistic: {arch_test[0]:.4f}")
print(f"p-value: {arch_test[1]:.4f}")
if arch_test[1] < 0.05:
    print("→ I residui mostrano eteroschedasticità (varianza non costante nel tempo).")
else:
    print("→ I residui NON mostrano evidenza di eteroschedasticità (varianza costante).")
"""

