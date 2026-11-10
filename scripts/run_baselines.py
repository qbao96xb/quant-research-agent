import json
from datetime import date
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from data.loader import fetch_ohlcv
from models.base import Forecast
from evaluation.walk_forward import walk_forward
from evaluation.metrics import score

ROOT = Path(__file__).resolve().parents[1]

TICKERS = ["MU", "KO"]
HORIZON = 5
START = -500          # must match run_kronos_wf.py
STEP = 5              # must match run_kronos_wf.py
LOOKBACK = 400
SEED = 1340           # baselines are deterministic, saved for the record


def _bdays(hist, horizon):
    last = pd.Timestamp(hist["timestamps"].iloc[-1])
    return pd.bdate_range(start=last + pd.offsets.BDay(1), periods=horizon)


class Naive:
    name = "naive"
    config = dict(model="naive")

    def forecast(self, hist, horizon):
        idx = _bdays(hist, horizon)
        m = np.full(horizon, hist["close"].iloc[-1])
        s = pd.Series(m, index=idx)
        return Forecast(index=idx, mean=s, lower=s * np.nan, upper=s * np.nan)


class Drift:
    name = "drift"
    config = dict(model="drift", lookback=LOOKBACK)

    def forecast(self, hist, horizon):
        idx = _bdays(hist, horizon)
        c = hist["close"].iloc[-LOOKBACK:].to_numpy()
        slope = (c[-1] - c[0]) / (len(c) - 1)
        s = pd.Series(c[-1] + slope * np.arange(1, horizon + 1), index=idx)
        return Forecast(index=idx, mean=s, lower=s * np.nan, upper=s * np.nan)


class Arima:
    name = "arima"
    config = dict(model="arima", order=(1, 1, 0), lookback=LOOKBACK)

    def forecast(self, hist, horizon):
        idx = _bdays(hist, horizon)
        c = hist["close"].iloc[-LOOKBACK:].to_numpy()
        fc = ARIMA(c, order=(1, 1, 0)).fit().get_forecast(horizon)
        ci = np.asarray(fc.conf_int(alpha=0.10))
        return Forecast(
            index=idx,
            mean=pd.Series(np.asarray(fc.predicted_mean), index=idx),
            lower=pd.Series(ci[:, 0], index=idx),
            upper=pd.Series(ci[:, 1], index=idx),
        )


data = {t: fetch_ohlcv(t) for t in TICKERS}

for fc in [Naive(), Drift(), Arima()]:
    print(f"\n=== {fc.name} ===")

    parts = []
    for t, df in data.items():
        print(f"--- {t} ---")
        res = walk_forward(df, fc, horizon=HORIZON, start=START, step=STEP)
        res.insert(0, "ticker", t)
        parts.append(res)
    res = pd.concat(parts, ignore_index=True)

    out = ROOT / "results" / fc.name / str(date.today())
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "predictions.csv", index=False)
    with open(out / "config.json", "w") as f:
        json.dump(dict(fc.config, tickers=TICKERS, horizon=HORIZON,
                       start=START, step=STEP, seed=SEED), f, indent=2, default=str)
    print(score(res))
    print("Saved to", out)
