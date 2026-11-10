"""Diebold-Mariano test on paired cumulative log-return errors.

Convention: d_t = L(e_A,t) - L(e_B,t), A = candidate, B = benchmark (Naive).
    Positive statistic: A has the larger loss (A worse).
    Negative statistic: A better.

Inference:
    - Newey-West (Bartlett-weighted HAC) long-run variance.
    - Harvey-Leybourne-Newbold small-sample correction.
    - Two-sided p-value from Student-t with n - 1 degrees of freedom.

Spec design: h=5, step=1 -> 4 HAC lags, Harvey with h_eff=5.
"""
from math import ceil
from numbers import Integral

import numpy as np
import pandas as pd
from scipy import stats

from evaluation.metrics import add_returns

MIN_PAIRS = 10


def _pos_int(name, v, minimum=1):
    if isinstance(v, bool) or not isinstance(v, Integral) or v < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def dm_test(eA, eB, h=5, loss="squared", alpha=0.05, step=1, hac_lags=None):
    """Diebold-Mariano test for two error Series indexed by unique origin.

    Parameters
    ----------
    h : forecast horizon in trading rows.
    step : trading rows between origins.
    hac_lags : HAC bandwidth in origin observations.
        None -> ceil(h / step) - 1 (mechanical overlap; spec: h - 1 at step=1).
    loss : "squared" or "absolute".

    Returns
    -------
    dict: dm, p, n, mean_diff, verdict, loss, step, hac_lags, h_eff.
    """
    _pos_int("h", h)
    _pos_int("step", step)
    if loss not in ("squared", "absolute"):
        raise ValueError("loss must be 'squared' or 'absolute'.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1.")

    h_eff = ceil(h / step)                  # effective horizon in origin units
    if hac_lags is None:
        hac_lags = h_eff - 1
    _pos_int("hac_lags", hac_lags, minimum=0)

    for s in (eA, eB):
        if s.index.has_duplicates or s.index.hasnans:
            raise ValueError("Error series need unique, non-missing origins.")

    paired = (pd.concat([eA.rename("A"), eB.rename("B")], axis=1, join="inner")
                .sort_index()
                .replace([np.inf, -np.inf], np.nan)
                .dropna())
    n = len(paired)
    out = dict(dm=np.nan, p=np.nan, n=n, mean_diff=np.nan,
               verdict="insufficient data", loss=loss, step=int(step),
               hac_lags=int(hac_lags), h_eff=h_eff)
    if n < MIN_PAIRS:
        return out
    if hac_lags >= n:
        raise ValueError("hac_lags must be smaller than the paired sample.")

    a, b = paired["A"].to_numpy(float), paired["B"].to_numpy(float)
    d = a * a - b * b if loss == "squared" else np.abs(a) - np.abs(b)
    out["mean_diff"] = float(d.mean())

    if not d.any():                         # identical forecasts (exact zeros)
        out.update(dm=0.0, p=1.0, verdict="no difference")
        return out

    # Scale for numerical stability (the statistic is scale-invariant).
    x = d / np.abs(d).max()
    xbar = x.mean()
    c = x - xbar
    lrv = c @ c / n
    for k in range(1, hac_lags + 1):
        lrv += 2 * (1 - k / (hac_lags + 1)) * (c[k:] @ c[:-k]) / n
    if not np.isfinite(lrv) or lrv <= 0:
        out["verdict"] = "undefined variance"
        return out

    corr = (n + 1 - 2 * h_eff + h_eff * (h_eff - 1) / n) / n
    if corr <= 0:
        out["verdict"] = "invalid small-sample correction"
        return out

    stat = float(xbar / np.sqrt(lrv / n) * np.sqrt(corr))
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    verdict = "no difference" if p >= alpha else ("A worse" if stat > 0 else "A better")
    out.update(dm=stat, p=p, verdict=verdict)
    return out


def _prep(res, h):
    """Errors and targets at horizon h, indexed by (ticker, origin)."""
    if "h" not in res.columns or "origin" not in res.columns:
        raise ValueError("Results need 'h' and 'origin' columns.")
    d = res.loc[res["h"] == h]               # filter first: validate fewer rows
    d = add_returns(d)
    if "ticker" not in d.columns:
        d = d.assign(ticker="all")
    if d.duplicated(["ticker", "origin"]).any():
        raise ValueError("Duplicate (ticker, origin) rows within a model.")
    return (d.set_index(["ticker", "origin"]).sort_index()
              [["target_date", "last", "actual_return", "err"]])


def dm_table(results, benchmark="naive", h=5, **kw):
    """DM test of every model against the benchmark, per ticker.

    results : dict[model name -> walk_forward output (with 'ticker')].
    **kw : forwarded to dm_test (step, hac_lags, loss, alpha).
        Pass the same `step` used in walk_forward.

    Returns DataFrame indexed by (ticker, model).
    """
    if benchmark not in results:
        raise ValueError(f"Missing benchmark: {benchmark}")

    base = _prep(results[benchmark], h)
    cols = ["ticker", "model", "h", "dm", "p", "n", "mean_diff",
            "verdict", "loss", "step", "hac_lags", "h_eff"]
    rows = []

    for name, res in results.items():
        if name == benchmark:
            continue
        j = _prep(res, h).join(base, how="inner", lsuffix="_A", rsuffix="_B")

        # Targets must be identical across models (vectorized over all tickers).
        same = (
            pd.to_datetime(j["target_date_A"]).eq(pd.to_datetime(j["target_date_B"])).all()
            and np.allclose(j["last_A"], j["last_B"], rtol=1e-12, atol=0)
            and np.allclose(j["actual_return_A"], j["actual_return_B"],
                            rtol=1e-12, atol=1e-14)
        )
        if not same:
            raise ValueError(f"Inconsistent evaluation targets for model: {name}")

        for ticker, g in j.groupby(level="ticker", sort=True):
            g = g.droplevel("ticker")
            rows.append(dict(ticker=ticker, model=name, h=h,
                             **dm_test(g["err_A"], g["err_B"], h=h, **kw)))

    return (pd.DataFrame(rows, columns=cols)
              .sort_values(["ticker", "model"], kind="stable")
              .set_index(["ticker", "model"]))