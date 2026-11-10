# src/tools/estimate.py
import json
import time
from pathlib import Path

import pandas as pd

from evaluation.walk_forward import _first_origin
from .config import MIN_HIST

ROOT = Path(__file__).resolve().parents[2]
CALIB = ROOT / "artifacts" / "calib.json"

# Seconds per fit. Starting guesses, refined by update_calib after each real run.
DEFAULT_SPO = {
    "naive": 0.003,
    "drift": 0.003,
    "arima": 0.07,
    "kronos_mini": 1.5,
    "kronos_small": 2.0,
    "kronos_base": 4.0,
    "kronos_small_ft": 2.0,
}
UNKNOWN_SPO = 0.25   # fallback for a model with no default or calibration

# refit_every per model (default 1). Must match FACTORIES in forecast_tools.py.
REFIT = {"arima": 1}

SAFETY = 1.2         # safety margin on the estimate


def _saved() -> dict:
    """Calibration saved by update_calib (empty if none)."""
    return json.loads(CALIB.read_text()) if CALIB.exists() else {}


def _spo() -> dict:
    """Seconds per fit: defaults, overridden by saved calibration."""
    return {**DEFAULT_SPO, **_saved()}


def update_calib(model: str, secs: float, n_fits: int) -> None:
    """Blend a new measurement into the saved seconds per fit (EWMA 0.7 / 0.3)."""
    if n_fits <= 0 or secs <= 0:
        return
    spo = _spo()
    obs = secs / n_fits
    d = _saved()                                  # save only measured values, not defaults
    d[model] = 0.7 * spo.get(model, obs) + 0.3 * obs
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    tmp = CALIB.with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    tmp.replace(CALIB)                            # atomic write


def n_origins(df: pd.DataFrame, start, horizon: int, step: int) -> int:
    """Number of forecast origins the backtest will use."""
    dates = pd.DatetimeIndex(pd.to_datetime(df["timestamps"]))
    first = max(_first_origin(dates, len(df), start), MIN_HIST - 1, 0)
    return max(len(range(first, len(df) - horizon, step)), 0)


def fits_per_model(model_name: str, n: int) -> float:
    """Number of model fits for n origins. No model is built, so Kronos weights stay unloaded."""
    return n / max(REFIT.get(model_name, 1), 1)


def estimate_s(df: pd.DataFrame, models: list, start, h: int, step: int) -> float:
    """Estimated runtime in seconds for the listed (uncached) models."""
    n = n_origins(df, start, h, step)
    spo = _spo()
    total = sum(fits_per_model(m, n) * spo.get(m, UNKNOWN_SPO) for m in models)
    return SAFETY * total
