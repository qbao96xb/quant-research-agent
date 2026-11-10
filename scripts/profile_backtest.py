"""Profile where walk_forward spends time, per model."""
import cProfile, pstats, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.loader import fetch_ohlcv
from models.baselines import Naive, Drift, Arima
from models.volatility import DEFAULT_VOL
from evaluation.walk_forward import walk_forward

# df = fetch_ohlcv("KO")
# START, STEP = "2024-06-01", 1     # ~400 origins
#
# for model in (Naive(), Arima()):
#     pr = cProfile.Profile()
#     t0 = time.time()
#     pr.runcall(walk_forward, df, model, horizon=5, start=START, step=STEP,
#                vol_model=DEFAULT_VOL)
#     dt = time.time() - t0
#     print(f"\n===== {model.name}: {dt:.1f}s =====")
#     pstats.Stats(pr).sort_stats("cumtime").print_stats(12)
import time
from tools.registry import call
t = time.time()
b = call("run_backtest", dataset_id=ds, start="2023-01-01", h=5, step=1)
print(time.time() - t)
