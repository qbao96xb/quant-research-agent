"""Chart layout, chart plans and extra chart types for chat findings.

findings.py imports this module as `V`. Functions here import findings lazily
(inside the function) to avoid a circular import.
"""
from math import ceil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# (normalised column name, legend label) for the p-value chart
PCOLS = (("dmpvalue", "DM vs naive"),
         ("cwpvalue", "CW vs naive (nested)"),
         ("dmpvsdrift", "DM vs drift"))

# request types, highest priority first, and the tools that signal them
ORDER = ("tune", "vol", "compare", "forecast", "track")
INTENT_TOOLS = {"tune": ("tune_arima",),
                "vol": ("garch_volatility",),
                "compare": ("compare_models", "run_backtest", "dm_test", "get_metrics"),
                "forecast": ("forecast", "forecast_path"),
                "track": ("forecast_vs_actual",)}
# table kind -> request type -> ordered chart names
PLANS = {"metrics": {"compare": ["pvalues", "scatter", "bar"],
                      "tune": ["pvalues", "scatter", "bar"]},
         "path": {"forecast": ["price_fan", "daily_bar"]}}
MAX_CHARTS = 4


def grid(n):
    """(rows, cols) for n panels: at most 3 columns; 4 panels -> 2x2."""
    if n < 1:
        raise ValueError("n must be >= 1")
    nc = 2 if n == 4 else min(3, n)
    return ceil(n / nc), nc


def intents(messages):
    """Request types of the current turn (tool calls after the last user message), in ORDER."""
    from . import findings as F
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    names = {name for i, name, _, _ in F._calls(messages) if i > last_user}
    return tuple(k for k in ORDER if names & set(INTENT_TOOLS[k]))


def plan(kind, question, intents=()):
    """Chart names for a table kind: the plan of the first matching intent plus triggered charts, capped."""
    from . import findings as F
    base = next((PLANS[kind][k] for k in intents if k in PLANS.get(kind, {})), None)
    if base is None:
        return F.pick_kinds(kind, question)
    q = question.lower()
    extra = [n for n, (_, ws) in F.CHARTS[kind].items() if F._hit(q, ws)]
    out = [n for n in dict.fromkeys(base + extra) if n in F.CHARTS[kind]]
    return out[:MAX_CHARTS]


def kind_of(df):
    """Table kinds handled here: 'tune' for the tune_arima order table, else None."""
    from . import findings as F
    m = F.cols(df)
    return "tune" if "order" in m and "validationrmse" in m else None


def tune_spec(df):
    from . import findings as F
    return dict(o=F.pick(df, "order", numeric=False), r=F.pick(df, "validationrmse"),
                tk=F.pick(df, "ticker", "symbol", numeric=False))


def metric_grid(df, s):
    """One bar panel per metric on a grid. Reference lines: DirAcc 0.5, Cov90 0.9."""
    from . import findings as F
    if not (s["ms"] and (s["md"] or s["tk"])):
        return None
    n = len(s["ms"])
    nr, nc = grid(n)
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.4 * nr), squeeze=False)
    flat = axes.ravel()
    for ax, y in zip(flat, s["ms"]):
        F._pivot(df, y, s["tk"], s["md"]).plot.bar(ax=ax, rot=30, title=y, legend=bool(s["tk"] and s["md"]))
        ax.set_xlabel("")
        if F.norm(y) in F.REFS:
            ax.axhline(F.REFS[F.norm(y)], color="red", ls="--", lw=1)
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig


def pvalues(df, s):
    """Grouped p-value bars per model (DM vs naive, CW vs naive, DM vs drift). Red line at 0.05."""
    from . import findings as F
    key = s["md"] or s["tk"]
    cols = [(F.pick(df, k), lab) for k, lab in PCOLS if F.pick(df, k)]
    if not (key and cols):
        return None
    if s["md"] and s["tk"]:
        idx = df[s["md"]].astype(str) + " (" + df[s["tk"]].astype(str) + ")"
    else:
        idx = df[key].astype(str)
    d = pd.DataFrame({lab: df[c].to_numpy(float) for c, lab in cols}, index=idx.to_numpy())
    d = d.dropna(how="all")
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    d.plot.bar(ax=ax, rot=30, width=0.8)
    ax.axhline(0.05, color="red", ls="--", lw=1)
    ax.set(ylim=(0, 1), xlabel="", ylabel="p-value",
           title="Significance vs benchmark (below the red line = significant)")
    ax.legend(fontsize=8)
    return fig


def tune_rmse(df, s):
    """Validation RMSE per ARIMA order, lowest first; the best order (per ticker) is green."""
    o, r, tk = s["o"], s["r"], s["tk"]
    if not (o and r):
        return None
    d = df.dropna(subset=[r]).sort_values(r).reset_index(drop=True)
    if d.empty:
        return None
    labels = d[o].astype(str)
    if tk and d[tk].nunique() > 1:
        labels = labels + " " + d[tk].astype(str)
        best = d.groupby(tk)[r].transform("min")
    else:
        best = pd.Series(d[r].min(), index=d.index)
    vals = d[r].to_numpy(float)
    colors = ["tab:green" if v == b else "lightgray" for v, b in zip(vals, best)]
    fig, ax = plt.subplots(figsize=(6, max(3.0, 0.32 * len(vals) + 1.2)))
    ax.barh(labels.to_numpy()[::-1], vals[::-1], color=colors[::-1])
    lo, hi = vals.min(), vals.max()
    pad = (hi - lo) * 0.5 or hi * 0.05
    ax.set_xlim(max(0.0, lo - pad), hi + pad)
    ax.set(xlabel="validation RMSE (cumulative log return)",
           title=f"ARIMA order search: best {labels.iloc[0]} (green)")
    return fig


def register(charts, default, spec, label):
    """Add the charts of this module to the findings registry (called once by findings.py)."""
    charts["metrics"]["pvalues"] = (pvalues, ("cw", "clark", "clark-west", "p-value", "p value",
                                                  "pvalue", "significance"))
    label["tune"] = "ARIMA order search"
    spec["tune"] = tune_spec
    charts["tune"] = {"tune_rmse": (tune_rmse, ("order", "tune", "tuning", "rmse"))}
    default["tune"] = ["tune_rmse"]
