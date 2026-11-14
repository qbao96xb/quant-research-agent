"""Order search for Arima on pre-test data. Pure functions, no registry.
"""
import numpy as np

from evaluation.dm_test import dm_test
from evaluation.metrics import errors
from evaluation.walk_forward import walk_forward
from models.baselines import Arima, Naive
from tools.config import ARIMA_CFG, MIN_HIST

SEARCH_CFG = ARIMA_CFG                            # same config as forecasts, for validation and test
MAX_ZERO_FRAC = 0.10                              # drop orders with >10% fallback forecasts
ALPHA = 0.05                                      # held-out check significance level



def _arima(order, cfg):
    return Arima(order=order, **cfg)              # fresh instance: the class keeps state



def _start(m, h, k):
    """walk_forward start giving exactly k origins with step=h on m rows."""
    return (m - h - 1) - (k - 1) * h



def make_splits(df, h, n_val, n_test):
    t_start = _start(len(df), h, n_test)
    val_df = df.iloc[: t_start + 1]               # last validation target = first test origin
    v_start = _start(len(val_df), h, n_val)
    if v_start < MIN_HIST - 1:
        raise ValueError("NOT_ENOUGH_DATA: history too short for this n_val/n_test")
    return (val_df, v_start), (df, t_start)



def _run(df, start, model, h, k):
    res = walk_forward(df, model, horizon=h, start=start, step=h,
                       min_hist=MIN_HIST, vol_model=None, terminal_only=True)
    e = errors(res, h)                            # Series indexed by origin
    if len(e) != k:
        raise ValueError(f"expected {k} origins, got {len(e)}")
    return e



def _rmse(e):
    return float(np.sqrt((e ** 2).mean()))



def _dm(eA, eB, h):
    r = dm_test(eA, eB, h=h, step=h)              # step=h -> no overlap, 0 HAC lags
    return {"p": r["p"], "dm": r["dm"], "verdict": r["verdict"], "n": r["n"]}



def held_out_check(dm_naive, dm_default):
    """'fail' if best is significantly worse than naive or ARIMA(1,0,1) on test.

    Reported only; never changes best_order. dm_default is None when best is (1,0,1).
    """
    bad = [f"{name} (DM p={d['p']:.3f})"
           for name, d in (("naive", dm_naive), ("ARIMA(1,0,1)", dm_default))
           if d is not None and d["verdict"] == "A worse" and d["p"] < ALPHA]
    if bad:
        return "fail", ("Best order is significantly worse on the held-out test than "
                        + " and ".join(bad) + ".")
    return "pass", ("Best order is not significantly worse than naive or ARIMA(1,0,1) "
                    + "on the held-out test.")



def tune_arima(df, h=5, max_p=2, max_q=2, n_val=60, n_test=40, cfg=None):
    cfg = dict(SEARCH_CFG if cfg is None else cfg)
    (vdf, vs), (tdf, ts) = make_splits(df, h, n_val, n_test)

    scores, zeros = {}, {}
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            o = (p, 0, q)
            m = _arima(o, cfg)
            scores[o] = _rmse(_run(vdf, vs, m, h, n_val))
            zeros[o] = m.config["n_zero_forecasts"]

    ok = {o: s for o, s in scores.items() if zeros[o] <= MAX_ZERO_FRAC * n_val}
    if not ok:
        raise RuntimeError("INTERNAL: every order failed to fit")
    best = min(ok, key=lambda o: (ok[o], sum(o)))   # ties -> simpler order

    # Held-out test: winner vs default vs naive. Never re-tune after this.
    e_best = _run(tdf, ts, _arima(best, cfg), h, n_test)
    e_def = _run(tdf, ts, _arima((1, 0, 1), cfg), h, n_test)
    e_nv = _run(tdf, ts, Naive(), h, n_test)
    dm_nv = _dm(e_best, e_nv, h)
    dm_def = None if best == (1, 0, 1) else _dm(e_best, e_def, h)
    check, reason = held_out_check(dm_nv, dm_def)

    ranked = sorted(scores.items(), key=lambda kv: (kv[1], sum(kv[0])))
    lines = ["| Order | Validation RMSE | Fallbacks |", "|---|---|---|"]
    lines += [f"| {o} | {s:.6f} | {zeros[o]} |" for o, s in ranked]

    return {
        "best_order": list(best),
        "table": "\n".join(lines),
        "test_rmse": {"best": _rmse(e_best), "default_101": _rmse(e_def), "naive": _rmse(e_nv)},
        "dm_best_vs_naive": dm_nv,
        "dm_best_vs_default": dm_def,
        "held_out_check": check,
        "check_reason": reason,
        "orders_tried": len(scores), "n_val": n_val, "n_test": n_test, "h": h, "step": h,
        "cfg": cfg,
        "note": (f"Selected from {len(scores)} orders on validation (n={n_val}, step={h}, "
                 f"non-overlapping). Test block (n={n_test}) was held out."),
    }
