# arima_forecast.py
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import bds
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

TICKER, START, END, H = "KO", "2015-01-01", "2026-10-30", 5
MAX_P, MAX_Q = 3, 3
ALPHA = 0.05          # 95% interval
LOOKBACK = 1000       # most recent daily returns used for fitting

# 1. Data -> log returns (in %, better for the optimizer) -------------------
px = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze().dropna()
r = 100 * np.log(px).diff().dropna().iloc[-LOOKBACK:]
last_price, last_date = float(px.iloc[-1]), px.index[-1]
print(f"{TICKER}: {len(r)} returns, last close {last_price:.4f} on {last_date.date()}")

# 2. Assumption checks ------------------------------------------------------
adf_p = adfuller(r, autolag="AIC")[1]
kpss_p = kpss(r, regression="c", nlags="auto")[1]
lb_p = acorr_ljungbox(r, lags=[10])["lb_pvalue"].iloc[0]
print(f"ADF p={adf_p:.4f} (want <0.05) | KPSS p={kpss_p:.4f} (want >0.05) | Ljung-Box p={lb_p:.4f}")
if adf_p > 0.05 or kpss_p < 0.05:
    print("WARNING: stationarity not supported, d=0 is questionable.")
if lb_p > 0.05:
    print("NOTE: returns look like white noise, so ARIMA will likely not beat naive.")

# 3. Order selection by BIC (in-sample, d=0) --------------------------------
rows = []
for p in range(MAX_P + 1):
    for q in range(MAX_Q + 1):
        try:
            f = ARIMA(r.to_numpy(), order=(p, 0, q)).fit()
            rows.append(dict(p=p, q=q, aic=f.aic, bic=f.bic))
        except Exception:
            pass
tab = pd.DataFrame(rows).sort_values(["bic", "p", "q"]).reset_index(drop=True)
print("\nTop 5 by BIC:\n", tab.head(5).to_string(index=False, float_format="%.2f"))
p, q = int(tab.p[0]), int(tab.q[0])
print(f"Selected ARIMA({p},0,{q})")

# 4. Final fit and residual check -------------------------------------------
fit = ARIMA(r.to_numpy(), order=(p, 0, q)).fit()
print("\n", fit.summary().tables[1])
res_lb = acorr_ljungbox(fit.resid, lags=[10])["lb_pvalue"].iloc[0]
print(f"Residual Ljung-Box p={res_lb:.4f} (want >0.05)")

# Linearity check
def linearity_checks(fit, max_dim=4):
    e = np.asarray(fit.resid)
    e = (e - e.mean()) / e.std()          # standardise before BDS

    lb = acorr_ljungbox(e, lags=[10])["lb_pvalue"].iloc[0]
    lb2 = acorr_ljungbox(e**2, lags=[10])["lb_pvalue"].iloc[0]
    stat, pval = bds(e, max_dim=max_dim)   # dims 2..max_dim
    print(f"Residual LB p={lb:.4f} | Squared-residual LB p={lb2:.4f}")
    for d, (s, p) in enumerate(zip(stat, pval), start=2):
        print(f"BDS dim={d}: stat={s:+.2f}, p={p:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].scatter(e[:-1], e[1:], s=6, alpha=0.5)
    ax[0].set(title="Residual vs lagged residual", xlabel="e[t-1]", ylabel="e[t]")
    plot_acf(e, lags=30, ax=ax[1], title="ACF of residuals")
    plot_acf(e**2, lags=30, ax=ax[2], title="ACF of squared residuals")
    plt.tight_layout()
    plt.show()

linearity_checks(fit)

# 5. Return forecast --------------------------------------------------------
fc = fit.get_forecast(steps=H)
mu = np.asarray(fc.predicted_mean)                  # daily mean returns, %
sd = np.asarray(fc.se_mean)                         # daily std, %
z = norm.ppf(1 - ALPHA / 2)

idx = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=H)   # weekdays only
daily = pd.DataFrame({"ret_%": mu, "lower_%": mu - z * sd, "upper_%": mu + z * sd}, index=idx)

# 6. Cumulative return and price forecast -----------------------------------
# Cumulative error at horizon h = sum of the first h forecast errors.
# With MA(inf) weights psi_j and S_k = sum_{j<=k} psi_j: Var = sigma2 * sum_{k<h} S_k^2
psi = arma2ma(np.r_[1, -fit.arparams], np.r_[1, fit.maparams], lags=H)
S = np.cumsum(psi)
sigma2 = fit.params[-1]
cum_mu = np.cumsum(mu) / 100                        # cumulative log return
cum_sd = np.sqrt(sigma2 * np.cumsum(S**2)) / 100

price = pd.DataFrame({
    "cum_ret_%": 100 * cum_mu,
    "price": last_price * np.exp(cum_mu),
    "lower": last_price * np.exp(cum_mu - z * cum_sd),
    "upper": last_price * np.exp(cum_mu + z * cum_sd),
}, index=idx)

print(f"\n== Daily log-return forecast (%), 95% interval ==\n{daily.to_string(float_format='%.4f')}")
print(f"\n== Price forecast (last close {last_price:.4f}) ==\n{price.to_string(float_format='%.4f')}")
