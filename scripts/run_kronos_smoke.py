from data.loader import fetch_ohlcv
from models.kronos_model import KronosForecaster

hist = fetch_ohlcv("MU")
fc = KronosForecaster(n_samples=5).forecast(hist, horizon=5)
print(fc.mean, fc.lower, fc.upper, sep="\n")
