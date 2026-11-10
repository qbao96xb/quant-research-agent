# src/tools/track_tools.py
import numpy as np
import pandas as pd

from evaluation.walk_forward import walk_forward
from .cache import get_dataset, get_meta, clean_json
from .config import MIN_HIST, make_vol
from .errors import ToolError
from .forecast_tools import make_model
from .registry_core import tool
from .schemas import TrackArgs
from .validation import check_horizon


def compute_track(df, forecaster, n, h, vol, min_hist=MIN_HIST):
    """One-step-ahead style track record: forecast each of the last n origins, compare to actual.

    Each forecast sees history up to its origin only (walk_forward guarantees this).
    Returns {"rows": [...], "summary": {...}}; returns are log returns over h days.
    """
    res = walk_forward(df, forecaster, horizon=h, start=-(n + h), step=1, min_hist=min_hist,
                       vol_model=vol, force_intervals=True, terminal_only=True)
    if res.empty:
        raise ToolError("NOT_ENOUGH_DATA", "Not enough history for this many origins.",
                        f"Use a smaller n (need at least {min_hist} rows of history).")
    last = res["last"].to_numpy(float)
    pred = np.log(res["pred"].to_numpy(float) / last)
    act = np.log(res["actual"].to_numpy(float) / last)
    lo = np.log(res["lower"].to_numpy(float) / last)
    hi = np.log(res["upper"].to_numpy(float) / last)
    err = pred - act
    inside = (act >= lo) & (act <= hi)
    rows = [dict(origin=str(o)[:10], date=str(t)[:10], pred_return=p, actual_return=a,
                 lo=l, hi=u, error=e, inside=bool(i))
            for o, t, p, a, l, u, e, i in zip(res["origin"], res["target_date"], pred, act, lo, hi, err, inside)]
    dir_acc = None if np.ptp(pred) < 1e-12 and abs(pred[0]) < 1e-12 else float(np.mean(np.sign(pred) == np.sign(act)))
    summary = dict(n=len(rows), hit_rate=float(inside.mean()), rmse=float(np.sqrt((err ** 2).mean())),
                   mae=float(np.abs(err).mean()), bias=float(err.mean()), dir_acc=dir_acc)
    return clean_json(dict(rows=rows, summary=summary))


@tool(TrackArgs, slow=True)
def forecast_vs_actual(args) -> dict:
    """Recent track record: predicted vs actual return for each of the last n origins (default 30).

    Use for "predicted vs actual", "how did the forecast do", or to judge a forecast's reliability.
    Each row: origin, date, pred_return, actual_return, lo/hi (90% interval), error, inside.
    `summary` has hit_rate (share of actuals inside the 90% interval; about 0.90 is calibrated),
    rmse, mae, bias, dir_acc (None for a zero-return forecast like naive).
    Walk-forward, no look-ahead. Slow for arima with large n; keep n at 30-60.
    """
    df = get_dataset(args.dataset_id)
    check_horizon(args.h, len(df))
    model = make_model(args.model, args.order)
    vol = make_vol(cached=True); vol.reset()
    out = compute_track(df, model, args.n, args.h, vol)
    return clean_json(dict(ok=True, ticker=get_meta(args.dataset_id).get("ticker"), model=args.model,
                           h=args.h, level="90%",
                           order=list(args.order) if args.order else None, **out))
