# src/report/facts.py
import math
import time

import numpy as np

from src.data.loader import fetch_ohlcv
from src.tools.registry import call
from src.tools.cache import get_dataset
from src.tools.config import make_vol

TRADING_DAYS = 252

# canonical key -> accepted aliases (all lowercase)
_METRIC_ALIASES = {
    "mae": ("mae", "mean_abs_error", "mean_absolute_error"),
    "rmse": ("rmse", "root_mean_squared_error"),
    "diracc": ("diracc", "dir_acc", "directional_accuracy", "hit_rate", "hit"),
    "cov90": ("cov90", "cov_90", "coverage", "coverage_90", "cov", "coverage90"),
    "width": ("width", "mean_width", "avg_width", "interval_width"),
}


def ok(r, step):
    """Return the tool result, or raise a readable RuntimeError."""
    if not r.get("ok"):
        e = r.get("error") or r
        raise RuntimeError(f"{step} failed: {e.get('code')} {e.get('message')}")
    return r


def _return_stats(ticker, start):
    """Mean and std of daily log returns from the cleaned price data.

    Returns {} on failure, so the report shows '-' instead of crashing.
    """
    try:
        df = fetch_ohlcv(ticker, start=start)
        r = np.log(df["close"].astype(float)).diff().dropna()
        return {
            "mean_ret": float(r.mean()),
            "vol_daily": float(r.std(ddof=1)),
        }
    except Exception:
        return {}


def _norm_describe(d):
    """Expose stable keys: start, end, last_date, n, vol_daily (keeps originals)."""
    d = dict(d or {})
    vol_1y = d.get("vol_1y_ann")
    vol_daily = (vol_1y / math.sqrt(TRADING_DAYS)
                 if isinstance(vol_1y, (int, float)) and math.isfinite(vol_1y) else None)
    return {
        **d,
        "start": d.get("start", d.get("first")),
        "end": d.get("end", d.get("last")),
        "last_date": d.get("last_date", d.get("last")),
        "n": d.get("n", d.get("rows")),
        "vol_daily": d.get("vol_daily", vol_daily),
    }


def _norm_metrics(r):
    """Lowercase keys, flatten a nested `metrics` dict, add canonical metric keys."""
    r = dict(r or {})
    inner = r.get("metrics")
    if isinstance(inner, dict):
        r = {**r, **inner}
    low = {str(k).lower(): v for k, v in r.items()}
    out = dict(low)
    for canon, aliases in _METRIC_ALIASES.items():
        out[canon] = next((low[a] for a in aliases if low.get(a) is not None), None)
    return out


def _norm_dm(r):
    """Accept `p` or `p_value`; always expose `p_value`."""
    return {**r, "p_value": r.get("p_value", r.get("p"))}


def _wait_backtest(bt, timeout=120, poll=2):
    """Poll backtest_status until done, or raise after `timeout` seconds."""
    t0 = time.time()
    while bt.get("status") == "running":
        if time.time() - t0 > timeout:
            raise RuntimeError(f"run_backtest timed out after {timeout}s")
        time.sleep(poll)
        bt = ok(call("backtest_status", backtest_id=bt["backtest_id"]), "backtest_status")
    return bt


def _history(ds, n=250):
    """Last n closes, for the price fan chart."""
    df = get_dataset(ds).iloc[-n:]
    return dict(dates=[str(t)[:10] for t in df["timestamps"]],
                close=[float(c) for c in df["close"]])


def _garch_diag(ds):
    """In-sample GARCH series for diagnostics. {} on failure (chart shows MISSING)."""
    try:
        return make_vol().diagnostics(get_dataset(ds))
    except Exception:
        return {}


def verdict(dm, alpha=0.05):
    out = {}
    for m, r in dm.items():
        p = r.get("p_value")
        if p is None or not math.isfinite(p):
            out[m] = f"No test result ({r.get('verdict', 'unavailable')})"
        elif p >= alpha:
            out[m] = f"No significant difference from naive (DM p={p:.3f})"
        else:
            direction = "better" if r.get("dm", 0) < 0 else "worse"
            out[m] = f"Significant difference from naive (DM p={p:.3f}, {m} {direction})"
    return out


def collect_facts(ticker, start="2015-01-01", h=5, models=("naive", "drift", "arima"),
                  bt_start=-500, bt_timeout=120):
    if "naive" not in models:
        raise ValueError("models must include 'naive' (the DM benchmark).")

    d = ok(call("load_data", ticker=ticker, start=start), "load_data")
    ds = d["dataset_id"]
    desc = _norm_describe(ok(call("describe_data", dataset_id=ds), "describe_data"))
    desc.update(_return_stats(ticker, start))   # mean_ret and exact daily vol

    fc = {m: ok(call("forecast", dataset_id=ds, model=m, h=h), f"forecast {m}")
          for m in models}

    bt = ok(call("run_backtest", dataset_id=ds, models=list(models),
                 start=bt_start, h=h, step=h), "run_backtest")
    bt = _wait_backtest(bt, timeout=bt_timeout)
    bid = bt["backtest_id"]

    met = {m: _norm_metrics(ok(call("get_metrics", backtest_id=bid, model=m), f"metrics {m}"))
           for m in models}
    dm = {m: _norm_dm(ok(call("dm_test", backtest_id=bid, model_a=m, model_b="naive"),
                         f"dm {m}"))
          for m in models if m != "naive"}
    vol = ok(call("garch_volatility", dataset_id=ds, h=h), "garch")
    paths = {m: ok(call("forecast_path", dataset_id=ds, model=m, h=h), f"path {m}")["path"]
             for m in models}

    facts = dict(ticker=ticker, start=start, h=h, step=h,
                 dataset_id=ds, backtest_id=bid,
                 n_origins=bt.get("origins"),
                 backtest_first=bt.get("first"), backtest_last=bt.get("last"),
                 describe=desc, forecast=fc, metrics=met, dm=dm, garch=vol,
                 paths=paths, history=_history(ds), garch_diag=_garch_diag(ds))
    facts["verdict"] = verdict(dm)
    return facts
