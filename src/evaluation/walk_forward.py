"""Walk-forward (rolling-origin) backtest.

At each origin t, the forecaster sees only rows 0..t, then predicts the next
`horizon` closes. Output is long format: one row per origin and h.

Columns: origin, h, target_date, last, actual, pred, lower, upper (prices).
Log returns are derived downstream in evaluation.metrics.

Example
-------
>>> res = walk_forward(df, Naive(), horizon=5, start="2023-01-01", step=1)
>>> res5 = res[res["h"] == 5]
"""
import numpy as np
import pandas as pd

from models.volatility import add_intervals, DEFAULT_VOL

COLUMNS = ["origin", "h", "target_date", "last", "actual",
           "pred", "lower", "upper"]


def _first_origin(dates, n, start):
    """Resolve `start` (int offset/position or date-like) to a row position."""
    if isinstance(start, (int, np.integer)) and not isinstance(start, bool):
        return n + int(start) if start < 0 else int(start)
    ts = pd.Timestamp(start)
    if dates.tz is not None and ts.tz is None:
        ts = ts.tz_localize(dates.tz)
    return int(dates.searchsorted(ts, side="left"))   # first date >= start


def split_at(df, t, horizon):
    """Return (hist, fut) for origin row t. hist ends at t, fut starts at t+1."""
    hist = df.iloc[: t + 1]
    fut = df.iloc[t + 1 : t + 1 + horizon]
    # No-look-ahead guard: every target date is strictly after the history.
    assert hist["timestamps"].iloc[-1] < fut["timestamps"].iloc[0], "look-ahead"
    return hist, fut


def walk_forward(df, forecaster, horizon=5, start="2023-01-01", step=1,
                 min_hist=400, vol_model=DEFAULT_VOL, force_intervals=False,
                 terminal_only=False):
    """Run a walk-forward backtest for one forecaster on one ticker.

    Parameters
    ----------
    df : DataFrame sorted by date with "timestamps" and "close" (> 0).
    forecaster : object with ``forecast(hist, horizon) -> Forecast``.
    horizon : int, default 5. Days ahead.
    start : str | Timestamp | int, default "2023-01-01"
        First origin. Date-like: first row on or after that date.
        Negative int: offset from the end. Non-negative int: row position.
        Use the same value for all models.
    step : int, default 1. Rows between origins.
        step < horizon gives overlapping targets, so the DM test needs HAC.
    min_hist : int, default 400. Minimum history rows; earlier origins skipped.
    vol_model : VolModel or None. Fills missing bounds. None disables.
    force_intervals : bool. True overwrites all bounds with vol_model bounds.
    terminal_only : bool. True keeps only h == horizon rows (smaller output).

    Returns
    -------
    DataFrame with COLUMNS (empty with the same schema if no origin qualifies).
    """
    if horizon < 1 or step < 1 or min_hist < 1:
        raise ValueError("horizon, step and min_hist must be >= 1.")
    for c in ("timestamps", "close"):
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    dates = pd.DatetimeIndex(pd.to_datetime(df["timestamps"]))
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Timestamps must be valid, unique and increasing.")
    px = df["close"].to_numpy(dtype=float)
    if not np.isfinite(px).all() or (px <= 0).any():
        raise ValueError("Close prices must be finite and positive.")

    n = len(df)
    first = max(_first_origin(dates, n, start), min_hist - 1, 0)
    hs = np.arange(1, horizon + 1)
    sel = slice(horizon - 1, horizon) if terminal_only else slice(None)

    chunks = {c: [] for c in COLUMNS}
    for t in range(first, n - horizon, step):         # last origin: n - horizon - 1
        hist, _ = split_at(df, t, horizon)
        fc = forecaster.forecast(hist, horizon)
        if vol_model is not None:
            fc = add_intervals(fc, hist, horizon, vol_model=vol_model,
                               force=force_intervals)
        pred = fc.mean.to_numpy(dtype=float)
        lower = fc.lower.to_numpy(dtype=float)
        upper = fc.upper.to_numpy(dtype=float)
        if not (len(pred) == len(lower) == len(upper) == horizon):
            raise ValueError("Forecast length must equal horizon.")

        chunks["origin"].append(np.repeat(dates[t], horizon)[sel])
        chunks["h"].append(hs[sel])
        chunks["target_date"].append(dates[t + 1 : t + 1 + horizon][sel])
        chunks["last"].append(np.repeat(px[t], horizon)[sel])
        chunks["actual"].append(px[t + 1 : t + 1 + horizon][sel])
        chunks["pred"].append(pred[sel])
        chunks["lower"].append(lower[sel])
        chunks["upper"].append(upper[sel])

    if not chunks["h"]:
        return pd.DataFrame(columns=COLUMNS)
    out = {}
    for c in COLUMNS:
        if c in ("origin", "target_date"):
            # Chunks may be ndarray or DatetimeIndex; normalize to datetime64 arrays.
            arr = np.concatenate([np.asarray(x, dtype="datetime64[ns]") for x in chunks[c]])
            out[c] = pd.DatetimeIndex(arr)
        else:
            out[c] = np.concatenate(chunks[c])
    return pd.DataFrame(out, columns=COLUMNS)
