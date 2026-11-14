# src/tools/forecast_tools.py
from pathlib import Path

from scipy.stats import norm
import numpy as np
from models.baselines import Naive, Drift, Arima
from models.volatility import add_intervals
from .config import ARIMA_CFG, KRONOS_SAMPLES, make_vol
from .vol_history import vol_history
from .kronos_cfg import get_kronos_cfg
from .cache import get_dataset, clean_json, get_meta
from .validation import check_origin, check_horizon
from .model_names import KRONOS_SIZES
from .registry_core import tool
from .schemas import ForecastArgs, GarchVolatilityArgs, ForecastPathArgs

def _kronos(size):
    from models.kronos_model import get_forecaster   # lazy import (torch, weights)
    c = get_kronos_cfg()
    return get_forecaster(size=size, n_samples=KRONOS_SAMPLES,
                          lookback=c["lookback"], T=c["T"])

def make_model(name, order=None):
    """Fresh instance per call for baselines. Kronos is cached by get_forecaster."""
    if order is not None:
        if name != "arima":
            raise ValueError("order is only valid for the arima model")
        return Arima(order=tuple(order), **ARIMA_CFG)
    return FACTORIES[name]()

FACTORIES = {
    "naive": lambda: Naive(),
    "drift": lambda: Drift(lookback=250),
    "arima": lambda: Arima(order=(1, 0, 1), **ARIMA_CFG),
    **{f"kronos_{s}": (lambda s=s: _kronos(s)) for s in KRONOS_SIZES},
}
VOL = make_vol()

def _z(level: float) -> float:
    return float(norm.ppf((1 + level) / 2))      # 0.90 -> 1.645, 0.95 -> 1.960

def _label(level: float) -> str:
    return f"{level * 100:g}%"

def _ticker(dataset_id):
    return get_meta(dataset_id).get("ticker")


def _pct(x: float) -> str:
    """0.00234795 -> '+0.23%'"""
    return f"{x * 100:+.2f}%"

def _to_returns(fc, last_close):
    f = lambda s: float(np.log(s.iloc[-1] / last_close))
    return f(fc.mean), f(fc.lower), f(fc.upper)

@tool(ForecastArgs)
def forecast(args) -> dict:
    """Forecast the h-day log return for one model at one date.

    Models: naive, drift, arima, kronos_mini, kronos_small, kronos_base.
    Use for a single forecast. For comparing models over time, use run_backtest.
    Returns the mean return and a GARCH-t interval (same for every model).
    Default level is 90%. Set `level` (e.g. 0.95) when the user asks for another level.
    Also returns mean_pct/lo_pct/hi_pct as formatted percentages; quote these in the answer.
    """
    hist = get_dataset(args.dataset_id)
    check_horizon(args.h, len(hist))
    hist = check_origin(hist, args.origin, args.h)
    fc = make_model(args.model, args.order).forecast(hist, args.h)
    fc = add_intervals(fc, hist, args.h, z=_z(args.level), vol_model=VOL, force=True)
    last_close = float(hist["close"].iloc[-1])
    mean, lo, hi = _to_returns(fc, last_close)
    price_mean = float(fc.mean.iloc[-1])
    return clean_json(dict(ok=True, model=args.model, h=args.h,
                           order=list(args.order) if args.order else ([1, 0, 1] if args.model == "arima" else None),
                           origin=hist["timestamps"].iloc[-1],
                           mean=mean, lo=lo, hi=hi,
                           price_mean=price_mean,
                           last_close=last_close,
                           horizon_days=args.h,
                           level=_label(args.level),
                           ticker=_ticker(args.dataset_id),
                           mean_pct=_pct(mean), lo_pct=_pct(lo), hi_pct=_pct(hi)))

@tool(GarchVolatilityArgs)
def garch_volatility(args) -> dict:
    """GARCH volatility forecast for each day 1..h from the last date.
    Returns per day: daily_sigma (1-day volatility of that day's return) and
    sigma (cumulative volatility of the return from the origin to that day),
    plus symmetric intervals at `level` (default 90%). Top-level `sigma` is the
    cumulative sigma at day h.
    Use for 'volatility for each day'. Quote daily_sigma_pct for daily values."""
    hist = get_dataset(args.dataset_id)
    check_horizon(args.h, len(hist))
    cum = np.atleast_1d(VOL.sigma(hist, args.h)).astype(float)
    if len(cum) != args.h:
        raise ValueError(f"expected {args.h} cumulative sigmas, got {len(cum)}")
    var = cum ** 2
    daily = np.sqrt(np.clip(np.diff(np.concatenate([[0.0], var])), 0, None))
    z, tk = _z(args.level), _ticker(args.dataset_id) or args.dataset_id
    path = [dict(ticker=tk, day=k + 1,
                 daily_sigma=daily[k], daily_sigma_pct=f"{daily[k] * 100:.2f}%",
                 sigma=cum[k], cum_sigma_pct=f"{cum[k] * 100:.2f}%",
                 lo=-z * cum[k], hi=z * cum[k])
            for k in range(args.h)]
    return clean_json(dict(ok=True, h=args.h, model=VOL.name, ticker=tk,
                           level=_label(args.level), sigma=cum[-1],
                           lo=-z * cum[-1], hi=z * cum[-1], path=path, history=vol_history(hist)))

@tool(ForecastPathArgs)
def forecast_path(args) -> dict:
    """Forecast the full daily path for days 1..h from one origin, in ONE call.

    Use this when the user wants a forecast for each day. Do NOT call `forecast`
    once per day. Returns, per day: price, cum_log_return (from origin),
    daily_log_return (day k vs day k-1), and a GARCH-t interval on the
    cumulative return (same for every model). Default level is 90%.
    Set `level` (e.g. 0.95) when the user asks for another level.

    Models: naive, drift, arima, kronos_mini, kronos_small, kronos_base, kronos_small_ft.
    """
    hist = get_dataset(args.dataset_id)
    check_horizon(args.h, len(hist))
    hist = check_origin(hist, args.origin, args.h)
    fc = make_model(args.model, args.order).forecast(hist, args.h)
    fc = add_intervals(fc, hist, args.h, z=_z(args.level), vol_model=VOL, force=True)

    n = len(fc.mean)
    if not (n == len(fc.lower) == len(fc.upper) == args.h):
        raise ValueError(f"path length mismatch: mean={n}, lower={len(fc.lower)}, "
                         f"upper={len(fc.upper)}, h={args.h}")

    last = float(hist["close"].iloc[-1])
    price = fc.mean.to_numpy(float)
    prev = np.concatenate([[last], price[:-1]])
    cum = np.log(price / last)
    daily = np.log(price / prev)
    lo = np.log(fc.lower.to_numpy(float) / last)
    hi = np.log(fc.upper.to_numpy(float) / last)

    tk = _ticker(args.dataset_id) or args.dataset_id  # fallback if metadata is missing
    path = [dict(ticker=tk, model=args.model, day=k + 1, price=price[k],
                 daily_pct=_pct(daily[k]), cum_pct=_pct(cum[k]),
                 cum_lo_pct=_pct(lo[k]), cum_hi_pct=_pct(hi[k]),
                 cum_log_return=cum[k], daily_log_return=daily[k],
                 lo=lo[k], hi=hi[k])
            for k in range(args.h)]
    return clean_json(dict(ok=True, model=args.model, h=args.h,
                           order=list(args.order) if args.order else ([1, 0, 1] if args.model == "arima" else None),
                           horizon_days=args.h, level=_label(args.level),
                           ticker=tk, origin=hist["timestamps"].iloc[-1],
                           last_close=last, path=path))


