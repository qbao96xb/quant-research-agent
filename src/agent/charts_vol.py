"""Conditional-volatility chart: recent in-sample vol (gray), the forecast (red), and a sigma note.

findings.py imports this module as `C`. Functions here import findings lazily
(inside the function) to avoid a circular import.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import charts_v2 as V

VOL_PLAN = ["vol_cond", "vol_fan", "vol_daily"]


def cond_vol(messages):
    """{TICKER: (DataFrame with date and cond_vol, model name)} from garch_volatility results."""
    from . import findings as F
    out = {}
    for _, name, _, data in F._calls(messages):
        if name != "garch_volatility" or not isinstance(data, dict) or not data.get("history"):
            continue
        tk = str(data.get("ticker") or "").upper()
        d = pd.DataFrame(data["history"])
        d["date"] = pd.to_datetime(d["date"])
        out[tk] = (d, str(data.get("model") or "GARCH"))
    return out


def sigma_note(tk, s1, close=None):
    """Plain-language meaning of a 1-day sigma s1 (decimal) at the last close."""
    pct = s1 * 100
    if close is None:
        return f"1-day sigma {pct:.2f}% for {tk} (about 68% of days fall within ±1 sigma if returns were normal)"
    a = close * np.exp(-s1)
    b = close * np.exp(s1)
    return (f"1-day sigma {pct:.2f}% for {tk} at last close ${close:.2f}: ±1 sigma is about "
            f"${a:.2f} to ${b:.2f} (about 68% of days if returns were normal)")


def vol_cond(df, s):
    """Gray: in-sample conditional daily vol; red: forecast daily sigma; sigma note under each panel."""
    from . import findings as F
    if not (s["t"] and s["d"]):
        return None
    cv = F.CTX.get("cond_vol", {})
    groups = list(df.groupby(s["tk"], sort=False)) if s["tk"] else [("", df)]
    groups = [(tk, d) for tk, d in groups if str(tk).upper() in cv]
    if not groups:
        return None
    fig, axes = plt.subplots(len(groups), 1, figsize=(7.5, 4.4 * len(groups)), squeeze=False)
    for ax, (tk, d) in zip(axes[:, 0], groups):
        key = str(tk).upper()
        hist, model = cv[key]
        d = d.sort_values(s["t"])
        last = hist["date"].iloc[-1]
        fdates = pd.bdate_range(last + pd.offsets.BDay(1), periods=len(d))
        daily = d[s["d"]].to_numpy(float)
        ax.plot(hist["date"], hist["cond_vol"] * 100, color="gray", lw=1.5, label="conditional vol (in-sample)")
        ax.plot(fdates, daily * 100, color="tab:red", marker="o", lw=1.8, label=f"{model} forecast")
        ax.axvline(last, color="black", ls="--", lw=0.8)
        ax.set(title=f"{key}: daily volatility", ylabel="daily sigma (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
        h = F.CTX.get("history", {}).get(key)
        close = float(h["close"].iloc[-1]) if h is not None and len(h) else None
        ax.text(0, -0.3, sigma_note(key, float(daily[0]), close), transform=ax.transAxes,
                fontsize=8, va="top")
    return fig


def register(charts, default, spec, label):
    """Add vol_cond to the findings registry and the vol plan (called once by findings.py)."""
    charts["vol"]["vol_cond"] = (vol_cond, ("conditional", "garch", "sigma", "volatility"))
    V.PLANS["vol"] = {"vol": list(VOL_PLAN)}
