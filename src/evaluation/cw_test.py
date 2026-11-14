"""Clark-West (2007) test for nested models on paired cumulative log-return errors.

Convention: A = larger candidate that nests B, B = benchmark (Naive).
    adj_t = eB_t^2 - (eA_t^2 - (fA_t - fB_t)^2)
    e = pred_return - actual_return, f = pred_return.
    Positive statistic: A has predictive gain over B.

Inference:
    - Newey-West (Bartlett-weighted HAC) long-run variance, same lags as dm_test.
    - One-sided p-value from the standard normal (H1: A better).
    - Valid only when A nests B (drift or ARIMA vs the zero forecast).
"""
from math import ceil
from numbers import Integral

import numpy as np
import pandas as pd
from scipy import stats

from evaluation.metrics import add_returns

MIN_PAIRS = 10
NESTED = ("drift", "arima")     # models that nest the zero (naive) forecast



def _pos_int(name, v, minimum=1):
    if isinstance(v, bool) or not isinstance(v, Integral) or v < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")



def _nw_lrv(x, hac_lags):
    """Bartlett-weighted Newey-West long-run variance of a 1-D array."""
    n = len(x)
    c = x - x.mean()
    lrv = c @ c / n
    for k in range(1, hac_lags + 1):
        lrv += 2 * (1 - k / (hac_lags + 1)) * (c[k:] @ c[:-k]) / n
    return lrv



def cw_test(eA, eB, fA, fB, h=5, alpha=0.05, step=1, hac_lags=None):
    """Clark-West MSPE-adjusted test for Series indexed by unique origin.

    eA, eB : forecast errors (pred_return - actual_return).
    fA, fB : predicted cumulative log returns.
    hac_lags : None -> ceil(h / step) - 1, as in dm_test.

    Returns dict: cw, p, n, mean_adj, verdict, hac_lags.
    """
    _pos_int("h", h)
    _pos_int("step", step)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    if hac_lags is None:
        hac_lags = ceil(h / step) - 1
    _pos_int("hac_lags", hac_lags, minimum=0)

    names = ("eA", "eB", "fA", "fB")
    series = (eA, eB, fA, fB)
    for s in series:
        if s.index.has_duplicates or s.index.hasnans:
            raise ValueError("Series need unique, non-missing origins.")

    paired = (pd.concat([s.rename(k) for s, k in zip(series, names)], axis=1, join="inner")
               .sort_index()
               .replace([np.inf, -np.inf], np.nan)
               .dropna())
    n = len(paired)
    out = dict(cw=np.nan, p=np.nan, n=n, mean_adj=np.nan,
               verdict="insufficient data", hac_lags=int(hac_lags))
    if n < MIN_PAIRS:
        return out
    if hac_lags >= n:
        raise ValueError("hac_lags must be smaller than the paired sample.")

    ea, eb, fa, fb = (paired[k].to_numpy(float) for k in names)
    adj = eb * eb - (ea * ea - (fa - fb) ** 2)
    out["mean_adj"] = float(adj.mean())

    if not adj.any():                       # identical forecasts (exact zeros)
        out.update(cw=0.0, p=0.5, verdict="no evidence")
        return out

    # Scale for numerical stability (the statistic is scale-invariant).
    x = adj / np.abs(adj).max()
    lrv = _nw_lrv(x, hac_lags)
    if not np.isfinite(lrv) or lrv <= 0:
        out["verdict"] = "undefined variance"
        return out

    stat = float(x.mean() / np.sqrt(lrv / n))
    p = float(stats.norm.sf(stat))
    out.update(cw=stat, p=p, verdict="A better" if p < alpha else "no evidence")
    return out



def _prep_cw(res, h):
    """Errors, predictions and targets at horizon h, indexed by (ticker, origin)."""
    if "h" not in res.columns or "origin" not in res.columns:
        raise ValueError("Results need 'h' and 'origin' columns.")
    d = add_returns(res.loc[res["h"] == h])
    if "ticker" not in d.columns:
        d = d.assign(ticker="all")
    if d.duplicated(["ticker", "origin"]).any():
        raise ValueError("Duplicate (ticker, origin) rows within a model.")
    return (d.set_index(["ticker", "origin"]).sort_index()
              [["target_date", "last", "actual_return", "pred_return", "err"]])



def cw_table(results, benchmark="naive", h=5, nested=NESTED, **kw):
    """Clark-West test of every nested model against the benchmark, per ticker.

    results : dict[model name -> walk_forward output (with 'ticker')].
    nested : model names tested; other models are skipped (CW needs nesting).
    **kw : forwarded to cw_test (step, hac_lags, alpha).
        Pass the same `step` used in walk_forward.

    Returns DataFrame indexed by (ticker, model)."""
    if benchmark not in results:
        raise ValueError(f"Missing benchmark: {benchmark}")

    base = _prep_cw(results[benchmark], h)
    cols = ["ticker", "model", "h", "cw", "p", "n", "mean_adj", "verdict", "hac_lags"]
    rows = []

    for name, res in results.items():
        if name == benchmark or name not in nested:
            continue
        j = _prep_cw(res, h).join(base, how="inner", lsuffix="_A", rsuffix="_B")

        same = (
            pd.to_datetime(j["target_date_A"]).eq(pd.to_datetime(j["target_date_B"])) .all()
            and np.allclose(j["last_A"], j["last_B"], rtol=1e-12, atol=0)
            and np.allclose(j["actual_return_A"], j["actual_return_B"],
                            rtol=1e-12, atol=1e-14)
        )
        if not same:
            raise ValueError(f"Inconsistent evaluation targets for model: {name}")

        for ticker, g in j.groupby(level="ticker", sort=True):
            g = g.droplevel("ticker")
            rows.append(dict(ticker=ticker, model=name, h=h,
                             **cw_test(g["err_A"], g["err_B"],
                                       g["pred_return_A"], g["pred_return_B"], h=h, **kw)))

    return (pd.DataFrame(rows, columns=cols)
              .sort_values(["ticker", "model"], kind="stable")
              .set_index(["ticker", "model"]))
