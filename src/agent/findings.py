import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import charts_v2 as V
from . import charts_hist as H
from . import charts_vol as C

ROOT = Path(__file__).resolve().parents[2]
ID_COLS = ("ticker", "symbol", "model", "method")
TIME = ("day", "date", "horizon", "step", "h", "ds", "timestamp")
METRICS = ("mae", "rmse", "diracc", "cov90", "width")
REFS = {"diracc": 0.5, "cov90": 0.9}          # reference lines on metric bars
LABEL = {"path": "Forecast path", "metrics": "Model comparison", "errors": "Forecast errors",
         "track": "Recent track record"}
CTX = {"history": {}, "level": 0.90}   # per-turn context for charts (set by write_findings)


# ---------- helpers ----------
def norm(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def cols(df):
    return {norm(c): c for c in df.columns}


def pick(df, *names, numeric=True):
    m = cols(df)
    for n in names:
        c = m.get(n)
        if c is not None and (not numeric or pd.api.types.is_numeric_dtype(df[c])):
            return c
    return None


def _num(s):
    s = str(s).strip()
    for ch in "%+,$*":
        s = s.replace(ch, "")
    try:
        return float(s)
    except ValueError:
        return s


def coerce(df):
    """Numeric copy for plotting. Text like '+1.4%' or '$95' becomes a number."""
    d = df.copy()
    for c in d.columns:
        if norm(c) in ID_COLS or pd.api.types.is_numeric_dtype(d[c]):
            continue
        v = pd.to_numeric(d[c].map(_num), errors="coerce")
        if v.notna().sum() >= max(1, len(v) // 2):
            d[c] = v
    return d


# ---------- parsing: tool calls ----------
def _args(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return {}
    return v if isinstance(v, dict) else {}


def _dig(obj, key, depth=0):
    if depth > 4:
        return None
    if isinstance(obj, dict):
        if isinstance(obj.get(key), (str, int)):
            return obj[key]
        for v in obj.values():
            r = _dig(v, key, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list) and obj:
        return _dig(obj[0], key, depth + 1)
    return None


def _calls(messages):
    """Yield (message index, tool name, arguments, parsed result) over the whole conversation."""
    pending = []
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                f = tc.get("function") or tc
                pending.append((tc.get("id"), f.get("name"), _args(f.get("arguments"))))
        elif role == "tool":
            cid = m.get("tool_call_id")
            k = next((j for j, p in enumerate(pending) if cid and p[0] == cid), 0 if pending else None)
            _, name, args = pending.pop(k) if k is not None else (None, None, {})
            name = name or m.get("name") or m.get("tool_name") or "tool"
            data = m.get("content")
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except ValueError:
                    pass
            yield i, name, args, data


def _load_dataset(dataset_id):
    try:
        from tools.cache import get_dataset
    except ImportError:
        from src.tools.cache import get_dataset
    return get_dataset(dataset_id)


def _history(messages, loader=None, n=60):
    """{TICKER: last n rows (timestamps, close)} for datasets loaded in this conversation."""
    loader = loader or _load_dataset
    out = {}
    for _, name, args, data in _calls(messages):
        if name != "load_data" or not isinstance(data, dict) or not args.get("ticker"):
            continue
        ds = _dig(data, "dataset_id")
        try:
            out[str(args["ticker"]).upper()] = loader(ds)[["timestamps", "close"]].tail(n).reset_index(drop=True)
        except Exception:
            continue
    return out


def _level(messages):
    """Interval level of the latest forecast result (e.g. '95%' -> 0.95). Default 0.90."""
    level = 0.90
    for _, name, _, data in _calls(messages):
        if name in ("forecast", "forecast_path") and isinstance(data, dict):
            try:
                level = float(str(data.get("level", "")).rstrip("%")) / 100
            except ValueError:
                pass
    return level if 0.5 <= level < 1 else 0.90


def _context(messages):
    ds, bt = {}, {}
    for _, name, args, data in _calls(messages):
        if not isinstance(data, dict):
            continue
        if name == "load_data" and args.get("ticker") and _dig(data, "dataset_id"):
            ds[_dig(data, "dataset_id")] = str(args["ticker"]).upper()
        if name == "run_backtest" and _dig(data, "backtest_id"):
            bt[_dig(data, "backtest_id")] = ds.get(args.get("dataset_id"))
    return ds, bt


def _frames(obj, depth=0):
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        yield pd.json_normalize(obj)
    elif isinstance(obj, dict) and depth < 3:
        if isinstance(obj.get("table"), str):          # compare_models: markdown table string
            yield from pipe_tables(obj["table"])
        vals = list(obj.values())
        if vals and all(isinstance(v, list) and v for v in vals) and len({len(v) for v in vals}) == 1:
            yield pd.DataFrame(obj)
        else:
            for v in vals:
                if isinstance(v, (list, dict)):
                    yield from _frames(v, depth + 1)


def tool_tables(messages):
    """(tool name, DataFrame) for tool results of the CURRENT turn, with ticker and model added."""
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    ds, bt = _context(messages)
    for i, name, args, data in _calls(messages):
        if i <= last_user or not isinstance(data, (dict, list)):
            continue
        ticker = (args.get("ticker") or _dig(data, "ticker")
                  or ds.get(args.get("dataset_id")) or bt.get(args.get("backtest_id")))
        model = args.get("model")
        for df in _frames(data):
            if model and not pick(df, "model", "method", numeric=False):
                df.insert(0, "model", model)
            if ticker and not pick(df, "ticker", "symbol", numeric=False):
                df.insert(0, "ticker", str(ticker).upper())
            yield name, df


# ---------- parsing: answer text (fallback) ----------
def pipe_tables(text):
    """Markdown pipe tables. The ticker comes from the title line above, e.g. 'KO' or 'DELL Forecast Data:'."""
    lines, i = text.splitlines(), 0
    n_tables = 0
    while i < len(lines):
        if ("|" in lines[i] and i + 1 < len(lines)
                and re.fullmatch(r"[\s|:\-+]+", lines[i + 1]) and "-" in lines[i + 1]):
            head = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            title = next((l.strip() for l in reversed(lines[:i]) if l.strip()), "")
            rows, i = [], i + 2
            while i < len(lines) and "|" in lines[i]:
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if len(cells) == len(head):
                    rows.append(cells)
                i += 1
            if rows:
                df = pd.DataFrame(rows, columns=head)
                m = re.match(r"\W*([A-Z]{1,6})\b", title)  # "KO", "KO's 7-day...", "DELL Forecast Data:"
                if not pick(df, "ticker", "symbol", numeric=False):
                    df.insert(0, "ticker", m.group(1) if m else f"table {n_tables}")
                n_tables += 1
                yield df
        else:
            i += 1


HEADER_RE = re.compile(r"^\s*(?:For\s+)?([A-Z]{1,6}):\s*$")
ROW_RE = re.compile(r"^\s*([A-Za-z][\w.\-]*):\s*(.+)$")
METRIC_RE = re.compile(r"([A-Za-z][A-Za-z0-9 \-]*?)\s+(-?\d+(?:\.\d+)?)\s*(?:,|$)")


def metric_lines(answer):
    """'For KO:' blocks with 'model: MAE 0.1, RMSE 0.2' lines."""
    rows, ticker = [], None
    for line in answer.splitlines():
        if (h := HEADER_RE.match(line)):
            ticker = h.group(1)
        elif ticker and (r := ROW_RE.match(line)):
            metrics = {k.strip(): float(v) for k, v in METRIC_RE.findall(r.group(2))}
            if len(metrics) >= 2:
                rows.append({"ticker": ticker, "model": r.group(1), **metrics})
    if rows:
        yield pd.DataFrame(rows)


def answer_tables(answer):
    for df in list(pipe_tables(answer)) + list(metric_lines(answer)):
        yield "answer", df

# ---------- per-origin errors (computed from the cached backtest, not via the LLM) ----------
def _wants_errors(question):
    q = question.lower()
    return any(_hit(q, CHARTS["errors"][n][1]) for n in CHARTS["errors"])


def error_frames(messages, max_n=250):
    """Long table (ticker, model, step, origin, error) for each backtest used in the CURRENT turn.
    error = log(pred/actual) = pred_return - actual_return, the same sign as score(). Positive = over-forecast."""
    try:
        from tools.cache import get as get_artifact       # CHECK: import path
    except ImportError:
        return []
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    _, bt_ticker = _context(messages)
    seen, out = set(), []
    for i, _, args, data in _calls(messages):
        if i <= last_user:
            continue
        bid = args.get("backtest_id") or _dig(data, "backtest_id")
        if not bid or bid in seen:
            continue
        seen.add(bid)
        try:
            bt = get_artifact(bid, "bt")
        except Exception:
            continue                                      # still running or missing
        for m, d in bt.groupby("model", sort=False):
            d = d.sort_values("origin").tail(max_n)
            err = np.log(d["pred"].to_numpy(float) / d["actual"].to_numpy(float))
            out.append(pd.DataFrame(dict(
                ticker=bt_ticker.get(bid) or str(d["ticker"].iloc[0]).upper(),
                model=m, step=range(1, len(d) + 1),
                origin=[str(pd.Timestamp(o).date()) for o in d["origin"]],
                error=err)))
    return [("errors", pd.concat(out, ignore_index=True))] if out else []

# ---------- classify and merge ----------
def err_col(df):
    return pick(df, "error", "abserror", "sqerror", "squarederror", "forecasterror", "loss")


def kind_of(df):
    m = cols(df)
    if "predreturn" in m and "actualreturn" in m:
        return "track"
    if "sigma" in m and any(k in m for k in TIME):
        return "vol"
    if "mae" in m or "rmse" in m:
        return "metrics"
    if "model" in m and err_col(coerce(df)):
        return "errors"
    if any(k in m for k in TIME) and any(
            k in m for k in ("cumlogreturn", "cumulativereturn", "cumreturn", "lo", "lower",
                             "lowerbound", "price", "mean")):
        return "path"
    return None


def merge(frames):
    """Group tables of the same kind and columns, so tickers share one figure."""
    groups = {}
    for _, df in frames:
        k = V.kind_of(df) or kind_of(df)
        if k:
            groups.setdefault((k, tuple(df.columns)), []).append(df)
    return [(k, pd.concat(p, ignore_index=True).drop_duplicates()) for (k, _), p in groups.items()]


# ---------- specs ----------
def path_spec(df):
    return dict(
        t=pick(df, *TIME),
        y=pick(df, "cumlogreturn", "cumulativereturn", "cumreturn", "mean"),
        lo=pick(df, "lo", "lower", "lowerbound", "cumlo"),
        hi=pick(df, "hi", "upper", "upperbound", "cumhi"),
        daily=pick(df, "dailylogreturn", "dailyreturn"),
        price=pick(df, "price"),
        tk=pick(df, "ticker", "symbol", numeric=False),
        md=pick(df, "model", "method", numeric=False))


def metric_spec(df):
    return dict(ms=[pick(df, m) for m in METRICS if pick(df, m)],
                p=pick(df, "dmpvalue", "pvalue", "dmp"),
                tk=pick(df, "ticker", "symbol", numeric=False),
                md=pick(df, "model", "method", numeric=False))


def error_spec(df):
    return dict(e=err_col(df), t=pick(df, "origin", "date", "t", "step", "h", "day"),
                tk=pick(df, "ticker", "symbol", numeric=False),
                md=pick(df, "model", "method", numeric=False))


SPEC = {"path": path_spec, "metrics": metric_spec, "errors": error_spec}


# ---------- layout helpers ----------
def _grid(df, key, w=5.5, h=3.6):
    groups = list(df.groupby(key, sort=False)) if key else [("", df)]
    n = len(groups)
    nc = min(n, 3)
    nr = math.ceil(n / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(w * nc, h * nr), squeeze=False)
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n], groups


def _series(d, md):
    return list(d.groupby(md, sort=False)) if md else [("", d)]


def _pivot(d, y, tk, md):
    if tk and md:
        return d.pivot_table(index=md, columns=tk, values=y)
    return d.set_index(md or tk)[y]


# ---------- path charts (forecast over the horizon) ----------
def line_ci(df, s):
    """Cumulative return line with a shaded 90% interval, one panel per ticker."""
    if not (s["t"] and s["y"]):
        return None
    fig, axes, groups = _grid(df, s["tk"])
    for ax, (name, d) in zip(axes, groups):
        for k, (label, sd) in enumerate(_series(d, s["md"])):
            sd = sd.sort_values(s["t"])
            line, = ax.plot(sd[s["t"]], sd[s["y"]], marker="o", label=str(label) or s["y"])
            if s["lo"] and s["hi"]:
                ax.fill_between(sd[s["t"]], sd[s["lo"]], sd[s["hi"]], color=line.get_color(),
                                alpha=0.15, label="interval" if k == 0 else None)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(title=str(name), xlabel=s["t"], ylabel=s["y"])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return fig


def whisker(df, s):
    """Point estimate with interval whiskers per day."""
    if not (s["t"] and s["y"] and s["lo"] and s["hi"]):
        return None
    fig, axes, groups = _grid(df, s["tk"])
    for ax, (name, d) in zip(axes, groups):
        for label, sd in _series(d, s["md"]):
            sd = sd.sort_values(s["t"])
            y = sd[s["y"]].to_numpy(float)
            err = [np.clip(y - sd[s["lo"]].to_numpy(float), 0, None),
                   np.clip(sd[s["hi"]].to_numpy(float) - y, 0, None)]
            ax.errorbar(sd[s["t"]], y, yerr=err, fmt="o", capsize=4, label=str(label) or s["y"])
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(title=f"{name} with interval".strip(), xlabel=s["t"], ylabel=s["y"])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return fig


def price_band(df, s):
    """Forecast price path. The band is rebuilt from the log-return interval."""
    if not (s["t"] and s["price"]):
        return None
    fig, axes, groups = _grid(df, s["tk"])
    log_ok = s["y"] and norm(s["y"]) == "cumlogreturn" and s["lo"] and s["hi"]
    for ax, (name, d) in zip(axes, groups):
        for label, sd in _series(d, s["md"]):
            sd = sd.sort_values(s["t"])
            line, = ax.plot(sd[s["t"]], sd[s["price"]], marker="o", label=str(label) or "price")
            if log_ok:
                p0 = (sd[s["price"]] / np.exp(sd[s["y"]])).iloc[0]
                ax.fill_between(sd[s["t"]], p0 * np.exp(sd[s["lo"]]), p0 * np.exp(sd[s["hi"]]),
                                color=line.get_color(), alpha=0.15)
        ax.set(title=str(name), xlabel=s["t"], ylabel="price")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return fig


def daily_bar(df, s):
    """Daily return per day as bars."""
    if not (s["t"] and s["daily"]):
        return None
    fig, axes, groups = _grid(df, s["tk"])
    for ax, (name, d) in zip(axes, groups):
        piv = (d.pivot_table(index=s["t"], columns=s["md"], values=s["daily"]) if s["md"]
               else d.set_index(s["t"])[[s["daily"]]])
        piv.plot.bar(ax=ax, rot=0, legend=bool(s["md"]), title=str(name))
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylabel(s["daily"])
    return fig


def overlay(df, s):
    """All tickers and models on one axis (cumulative return)."""
    if not (s["t"] and s["y"]):
        return None
    keys = [c for c in (s["tk"], s["md"]) if c]
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, d in (df.groupby(keys, sort=False) if keys else [("", df)]):
        d = d.sort_values(s["t"])
        label = " ".join(label) if isinstance(label, tuple) else str(label)
        ax.plot(d[s["t"]], d[s["y"]], marker="o", label=label or s["y"])
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(xlabel=s["t"], ylabel=s["y"])
    ax.grid(alpha=0.3)
    ax.legend()
    return fig

# ---------- volatility charts (forecast over the horizon) ----------
def vol_spec(df):
    return dict(t=pick(df, *TIME), s=pick(df, "daily_sigma", "sigma"), d=pick(df, "dailysigma"),
                lo=pick(df, "lo"), hi=pick(df, "hi"),
                tk=pick(df, "ticker", "symbol", numeric=False))

def vol_fan(df, s):
    """Cumulative GARCH sigma (%) by day, with the +/- z*sigma band."""
    if not (s["t"] and s["s"]):
        return None
    fig, axes, groups = _grid(df, s["tk"])
    for ax, (name, d) in zip(axes, groups):
        d = d.sort_values(s["t"])
        ax.plot(d[s["t"]], d[s["s"]] * 100, marker="o", label="sigma")
        if s["lo"] and s["hi"]:
            ax.fill_between(d[s["t"]], d[s["lo"]] * 100, d[s["hi"]] * 100,
                            alpha=0.15, label="interval")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(title=str(name), xlabel=s["t"], ylabel="cumulative sigma (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return fig

# ---------- evaluation charts (backtest metrics) ----------
def metric_bar(df, s):
    """One bar panel per metric. Reference lines: DirAcc 0.5, Cov90 0.9."""
    if not (s["ms"] and (s["md"] or s["tk"])):
        return None
    fig, axes = plt.subplots(1, len(s["ms"]), figsize=(3.8 * len(s["ms"]), 3.8), squeeze=False)
    for ax, y in zip(axes[0], s["ms"]):
        _pivot(df, y, s["tk"], s["md"]).plot.bar(ax=ax, rot=30, title=y, legend=bool(s["tk"] and s["md"]))
        ax.set_xlabel("")
        if norm(y) in REFS:
            ax.axhline(REFS[norm(y)], color="red", ls="--", lw=1)
    return fig


def dm_chart(df, s):
    """DM p-value per model. Bars below the red line (0.05) are significant vs naive."""
    if not (s["p"] and (s["md"] or s["tk"])):
        return None
    d = df.dropna(subset=[s["p"]])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    _pivot(d, s["p"], s["tk"], s["md"]).plot.bar(ax=ax, rot=30, legend=bool(s["tk"] and s["md"]))
    ax.axhline(0.05, color="red", ls="--", lw=1)
    ax.set(ylim=(0, 1), xlabel="", ylabel="DM p-value",
           title="DM p-value vs naive (below the red line = significant)")
    return fig


def rel_rmse(df, s):
    """RMSE divided by the naive RMSE. Below 1 means lower error than naive."""
    y = pick(df, "rmse")
    if not (y and s["md"]):
        return None
    base = df[df[s["md"]] == "naive"]
    if base.empty:
        return None
    d = df[df[s["md"]] != "naive"].copy()
    if s["tk"]:
        d["rel"] = d[y] / d[s["tk"]].map(base.set_index(s["tk"])[y])
    else:
        d["rel"] = d[y] / base[y].iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    _pivot(d, "rel", s["tk"], s["md"]).plot.bar(ax=ax, rot=30, legend=bool(s["tk"]))
    ax.axhline(1, color="red", ls="--", lw=1)
    ax.set(xlabel="", ylabel="RMSE / naive RMSE", title="RMSE relative to naive (below 1 = lower error)")
    return fig


def cov_width(df, s):
    """Interval calibration: coverage vs width. A wide interval covers easily."""
    c, w = pick(df, "cov90"), pick(df, "width")
    if not (c and w):
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, d in (df.groupby(s["tk"], sort=False) if s["tk"] else [("", df)]):
        ax.scatter(d[w], d[c], s=50, label=str(label) or None)
        if s["md"]:
            for _, r in d.iterrows():
                ax.annotate(str(r[s["md"]]), (r[w], r[c]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.9, color="red", ls="--", lw=1)
    ax.set(xlabel="Width", ylabel="Cov90", title="Coverage vs width (target 0.90)")
    ax.grid(alpha=0.3)
    if s["tk"]:
        ax.legend()
    return fig


def heatmap(df, s):
    """Metrics table as a heatmap. Colors are scaled per column, numbers are the raw values."""
    if not (s["ms"] and (s["md"] or s["tk"])):
        return None
    labels = (df[s["tk"]].astype(str) + " " + df[s["md"]].astype(str)) if (s["tk"] and s["md"]) \
        else df[s["md"] or s["tk"]].astype(str)
    vals = df[s["ms"]].to_numpy(float)
    lo, hi = np.nanmin(vals, 0), np.nanmax(vals, 0)
    z = np.nan_to_num((vals - lo) / np.where(hi - lo == 0, 1, hi - lo))
    fig, ax = plt.subplots(figsize=(1.3 * len(s["ms"]) + 3, 0.5 * len(df) + 1.5))
    ax.imshow(z, cmap="Blues", aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.3g}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(s["ms"])))
    ax.set_xticklabels(s["ms"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Metrics (color scaled per column)")
    return fig


# ---------- error charts (need raw per-origin errors) ----------
def error_box(df, s):
    """Box and whisker plot of forecast errors per model."""
    fig, axes, groups = _grid(df, s["tk"], w=4.5)
    for ax, (name, d) in zip(axes, groups):
        parts = [(str(m), g[s["e"]].dropna()) for m, g in d.groupby(s["md"], sort=False)]
        ax.boxplot([p[1] for p in parts], showmeans=True)
        ax.set_xticks(range(1, len(parts) + 1))
        ax.set_xticklabels([p[0] for p in parts], rotation=30)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(title=f"{name} {s['e']}".strip())
        ax.grid(alpha=0.3)
    return fig


def error_hist(df, s):
    fig, axes, groups = _grid(df, s["tk"], w=4.5)
    for ax, (name, d) in zip(axes, groups):
        for m, g in d.groupby(s["md"], sort=False):
            ax.hist(g[s["e"]].dropna(), bins=20, alpha=0.5, label=str(m))
        ax.set(title=f"{name} {s['e']}".strip())
        ax.legend(fontsize=8)
    return fig


def cum_error(df, s):
    """Cumulative squared error over origins. A lower, flatter line is better."""
    if not s["t"]:
        return None
    fig, axes, groups = _grid(df, s["tk"])
    sq = norm(s["e"]) in ("error", "forecasterror")
    for ax, (name, d) in zip(axes, groups):
        for m, g in d.sort_values(s["t"]).groupby(s["md"], sort=False):
            e = g[s["e"]] ** 2 if sq else g[s["e"]]
            ax.plot(g[s["t"]], e.cumsum(), label=str(m))
        ax.set(title=str(name), xlabel=s["t"], ylabel="cumulative loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return fig



# ---------- track record (predicted vs actual, recent origins) ----------
def track_spec(df):
    return dict(t=pick(df, "date", "targetdate", "origin", numeric=False), p=pick(df, "predreturn"),
                a=pick(df, "actualreturn"), lo=pick(df, "lo", "lower"), hi=pick(df, "hi", "upper"),
                tk=pick(df, "ticker", "symbol", numeric=False),
                md=pick(df, "model", "method", numeric=False))


def _panel_frame(df, s):
    """Add a one-column panel key: 'TICKER model'."""
    d = df.copy()
    keys = [c for c in (s["tk"], s["md"]) if c]
    d["_panel"] = d[keys].astype(str).agg(" ".join, axis=1) if keys else ""
    return d


def track_bars(df, s):
    """Predicted vs actual return per day (paired bars), interval whiskers, breaches in red."""
    if not (s["t"] and s["p"] and s["a"] and s["lo"] and s["hi"]):
        return None
    d = _panel_frame(df, s)
    fig, axes, groups = _grid(d, "_panel", w=7.5)
    for ax, (name, g) in zip(axes, groups):
        g = g.sort_values(s["t"]).reset_index(drop=True)
        x = np.arange(len(g)); p, a = g[s["p"]].to_numpy(float), g[s["a"]].to_numpy(float)
        lo, hi = g[s["lo"]].to_numpy(float), g[s["hi"]].to_numpy(float)
        out = (a < lo) | (a > hi)
        ax.bar(x - 0.2, p, 0.4, color="tab:blue", label="predicted")
        ax.bar(x + 0.2, a, 0.4, color="0.55", label="actual")
        ax.errorbar(x - 0.2, p, yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)],
                    fmt="none", ecolor="tab:blue", elinewidth=0.8, capsize=2, alpha=0.7)
        ax.scatter(x[out] + 0.2, a[out], s=28, color="tab:red", zorder=3, label="outside interval")
        ax.axhline(0, color="gray", lw=0.8)
        step = max(1, len(g) // 10)
        ax.set_xticks(x[::step]); ax.set_xticklabels([str(v)[5:10] if "-" in str(v) else str(v) for v in g[s["t"]]][::step],
                                                   rotation=45, fontsize=7)
        ax.set(title=f"{name}: hit rate {1 - out.mean():.0%} ({int((~out).sum())}/{len(g)} inside interval)",
               ylabel="return")
        ax.legend(fontsize=7, ncol=3, loc="upper left")
        ax.grid(alpha=0.3)
    return fig


def track_scatter(df, s, min_n=20):
    """Mincer-Zarnowitz: actual vs predicted with 45 degree line. Needs >= min_n points."""
    if not (s["p"] and s["a"]):
        return None
    from scipy import stats
    d = _panel_frame(df, s)
    groups = [(n, g) for n, g in d.groupby("_panel", sort=False) if len(g) >= min_n]
    if not groups:
        return None
    nc = min(len(groups), 3); nr = math.ceil(len(groups) / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 4 * nr), squeeze=False)
    flat = list(axes.flat)
    for ax in flat[len(groups):]:
        ax.set_visible(False)
    for ax, (name, g) in zip(flat, groups):
        x, y = g[s["p"]].to_numpy(float), g[s["a"]].to_numpy(float)
        lim = float(np.nanmax(np.abs(np.r_[x, y]))) * 1.05
        ax.scatter(x, y, s=12, alpha=0.6)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="45 degrees (perfect)")
        title = str(name)
        if np.std(x) > 0:
            r = stats.linregress(x, y); xs = np.array([x.min(), x.max()])
            ax.plot(xs, r.intercept + r.slope * xs, "r-", lw=1.2, label="fit")
            title += f"\nslope={r.slope:.2f}, a={r.intercept:.4f}, R2={r.rvalue ** 2:.3f}"
        else:
            title += "\nconstant forecast: no slope"
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="predicted return", ylabel="actual return")
        ax.set_title(title, fontsize=9); ax.legend(fontsize=7)
    return fig


SPEC["track"] = track_spec


# ---------- fan chart with history ----------
def price_fan(df, s):
    """Recent price history plus the forecast path with nested prediction bands.
    Falls back to price_band when the history is not available."""
    from scipy import stats
    if not (s["t"] and s["price"] and s["y"] and s["lo"] and s["hi"]):
        return None
    hist = CTX.get("history", {})
    keys = {str(k) for k in (df[s["tk"]].unique() if s["tk"] else [""])}
    if not keys & set(hist):
        return price_band(df, s)
    fig, axes, groups = _grid(df, s["tk"], w=7)
    level = CTX.get("level", 0.90)
    z_out = stats.norm.ppf((1 + level) / 2)
    for ax, (name, d) in zip(axes, groups):
        h = hist.get(str(name))
        if h is None:
            continue
        hd = pd.to_datetime(h["timestamps"]); last = float(h["close"].iloc[-1])
        ax.plot(hd, h["close"], color="k", lw=1.1, label="close")
        for k, (label, sd) in enumerate(_series(d, s["md"])):
            sd = sd.sort_values(s["t"]); n = len(sd)
            x = [hd.iloc[-1], *pd.bdate_range(hd.iloc[-1], periods=n + 1)[1:]]
            cum = sd[s["y"]].to_numpy(float)
            line, = ax.plot(x, np.r_[last, sd[s["price"]].to_numpy(float)], marker="o", ms=3,
                            label=str(label) or "forecast")
            if k == 0:                                           # bands around the first model
                lo, hi = sd[s["lo"]].to_numpy(float), sd[s["hi"]].to_numpy(float)
                for lv in sorted([l for l in (0.5, 0.8) if l < level] + [level], reverse=True):
                    r = stats.norm.ppf((1 + lv) / 2) / z_out
                    ax.fill_between(x, last * np.exp(np.r_[0, cum + (lo - cum) * r]),
                                    last * np.exp(np.r_[0, cum + (hi - cum) * r]),
                                    color=line.get_color(), alpha=0.18, lw=0, label=f"{lv:.0%} band")
        ax.axvline(hd.iloc[-1], color="gray", ls=":", lw=0.8)
        ax.set(title=f"{name}: price history and forecast fan", ylabel="price")
        ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=7, ncol=3, loc="upper left")
        ax.grid(alpha=0.3)
    return fig


def vol_daily(df, s):
    """Daily GARCH sigma (%) per forecast day, as bars."""
    if not (s["t"] and s["d"]):
        return None
    fig, axes, groups = _grid(df, s["tk"])
    for ax, (name, g) in zip(axes, groups):
        g = g.sort_values(s["t"])
        ax.bar(g[s["t"]].astype(int), g[s["d"]] * 100, color="tab:orange")
        ax.set(title=f"{name}: forecast daily volatility".strip(), xlabel=s["t"], ylabel="daily sigma (%)")
        ax.grid(alpha=0.3, axis="y")
    return fig

# ---------- registry: chart name -> (function, trigger words) ----------
CHARTS = {
    "path": {
        "price_fan": (price_fan, ("fan", "history", "price")),
        "line_ci": (line_ci, ("line", "interval", "ci", "band", "confidence", "fan")),
        "whisker": (whisker, ("whisker", "error bar", "errorbar")),
        "price": (price_band, ("price",)),
        "daily_bar": (daily_bar, ("bar", "bars", "histogram", "daily bar")),
        "overlay": (overlay, ("overlay", "together", "same chart", "same plot", "compare")),
    },
    "metrics": {
        "bar": (V.metric_grid, ("bar", "bars", "compare", "comparison", "rank")),
        "dm": (dm_chart, ("dm", "significance", "significant", "p-value", "p value")),
        "rel_rmse": (rel_rmse, ("relative", "ratio", "vs naive", "versus naive")),
        "scatter": (cov_width, ("coverage", "width", "scatter", "calibration")),
        "heatmap": (heatmap, ("heatmap", "heat map", "table")),
    },
    "errors": {
        "box": (error_box, ("box", "boxplot", "whisker", "distribution")),
        "hist": (error_hist, ("hist", "histogram", "density")),
        "cum_error": (cum_error, ("cumulative", "over time")),
    },
}
DEFAULT = {"path": ["price_fan", "line_ci", "daily_bar"],
           "metrics": ["bar", "dm", "rel_rmse", "scatter"],
           "errors": ["box", "hist", "cum_error"]}

# volatility charts
LABEL["vol"] = "Volatility"
SPEC["vol"] = vol_spec
CHARTS["vol"] = {"vol_fan": (vol_fan, ("volatility", "vol", "risk", "garch", "sigma")),
                 "vol_daily": (vol_daily, ("daily", "each day", "bar", "bars"))}
DEFAULT["vol"] = ["vol_fan", "vol_daily"]

# track record charts
CHARTS["track"] = {"track_bars": (track_bars, ("predicted", "actual", "track", "evaluat", "bar", "bars")),
                   "track_scatter": (track_scatter, ("scatter", "calibration", "mincer", "slope"))}
DEFAULT["track"] = ["track_bars", "track_scatter"]
V.register(CHARTS, DEFAULT, SPEC, LABEL)
C.register(CHARTS, DEFAULT, SPEC, LABEL)

def _hit(q, words):
    return any(re.search(rf"\b{re.escape(w)}\b", q) for w in words)


def pick_kinds(kind, question):
    q = question.lower()
    hits = [n for n, (_, ws) in CHARTS[kind].items() if _hit(q, ws)]
    return hits or DEFAULT[kind]


def make_charts(kind, df, title, question, out, idx):
    """Returns (chart paths, warnings). Failures are reported, not hidden."""
    num = coerce(df)
    spec = SPEC[kind](num)
    paths, warns = [], []
    for name in V.plan(kind, question, CTX.get("intents", ())):
        try:
            fig = CHARTS[kind][name][0](num, spec)
        except Exception as e:
            warns.append(f"Chart `{name}` failed: {type(e).__name__}: {e}")
            plt.close("all")
            continue
        if fig is None:
            warns.append(f"Chart `{name}` skipped: the data lacks the columns it needs.")
            continue
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p = out / f"chart_{idx:02d}_{name}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths, warns


# ---------- report ----------
def _display(df):
    m = cols(df)
    if "origin" in m and "error" in m:                    # error table: show a summary, CSV keeps all rows
        keys = [c for c in ("ticker", "model") if c in df.columns]
        return (df.groupby(keys, sort=False)["error"]
                  .agg(n="size", bias="mean",
                       MAE=lambda e: e.abs().mean(),
                       RMSE=lambda e: np.sqrt((e ** 2).mean()))
                  .reset_index())
    if "cumpct" in m:
        df = df.drop(columns=[m[k] for k in ("cumlogreturn", "dailylogreturn", "lo", "hi") if k in m])
    return df

def write_findings(result, question=""):
    """Returns (md_path, tables_md, chart_paths) for one run_agent result."""
    run_id = str(result["run_id"])
    out = ROOT / "results" / "chat" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "messages_debug.json").write_text(json.dumps(result["messages"], default=str, indent=1))

    head = [f"# Findings: {question or run_id}", "",
            f"> run `{run_id}`, status `{result['status']}`", "",
            "## Answer", "", result["answer"], ""]
    CTX["history"] = _history(result["messages"], n=H.history_len(result["messages"]))
    CTX["level"] = _level(result["messages"])
    CTX["intents"] = V.intents(result["messages"])
    CTX["cond_vol"] = C.cond_vol(result["messages"])
    frames = list(tool_tables(result["messages"]))
    if _wants_errors(question):
        frames += error_frames(result["messages"])
    merged = merge(frames) or merge(list(answer_tables(result["answer"])))

    body, charts = [], []
    for i, (kind, df) in enumerate(merged, 1):
        tk = pick(df, "ticker", "symbol", numeric=False)
        title = LABEL[kind] + (": " + ", ".join(map(str, df[tk].unique())) if tk else "")
        df.to_csv(out / f"table_{i:02d}.csv", index=False)
        body += [f"### {i}. {title}", "", _display(df).round(4).to_markdown(index=False), ""]
        paths, warns = make_charts(kind, df, title, question, out, i)
        for p in paths:
            charts.append(p)
            body += [f"![{p.stem}]({p.name})", ""]
        body += [f"*{w}*" for w in warns] + ([""] if warns else [])

    kinds = {k for k, _ in merged}
    q = question.lower()
    if "errors" not in kinds and any(_hit(q, CHARTS["errors"][n][1]) for n in ("box", "hist", "cum_error")):
        body += ["*Box, histogram and cumulative-error charts need per-origin errors. "
                 "The current tools return summary metrics only.*", ""]

    last_user = max((i for i, m in enumerate(result["messages"]) if m.get("role") == "user"), default=-1)
    for i, name, _, data in _calls(result["messages"]):          # report figures from make_report
        if name == "make_report" and i > last_user and isinstance(data, dict):
            for rel in data.get("figures", []):
                if (ROOT / rel).exists():
                    charts.append(ROOT / rel)

    text = "\n".join(body) if body else "*No tabular data found this turn.*"
    md = out / "findings.md"
    md.write_text("\n".join(head + ["## Tables and charts", "", text, ""]))
    tables_md = "\n".join(l for l in body if not l.startswith("!["))
    return md, tables_md, charts
