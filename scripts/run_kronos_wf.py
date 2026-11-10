import json
from datetime import date
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from data.loader import fetch_ohlcv
from models.kronos_model import get_forecaster
from evaluation.walk_forward import walk_forward
from evaluation.metrics import score

ROOT = Path(__file__).resolve().parents[1]   # scripts/ -> project root

TICKERS = ["MU", "KO"]
START = -500
STEP = 5
HORIZON = 5
N_SAMPLES = 10
SEED = 1340

VARIANTS = [
    ("small", 30),
    ("small", 60),
    ("small", 90),
    ("small_ft", 90),
]

data = {t: fetch_ohlcv(t) for t in TICKERS}
summary = {}

for size, lb in VARIANTS:
    fc = get_forecaster(size, n_samples=N_SAMPLES, lookback=lb, seed=SEED)
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

    summary[fc.name] = score(res)
    print(summary[fc.name])
    print("Saved to", out)

print("\n=== Summary ===")
for name, s in summary.items():
    print(name, s)
