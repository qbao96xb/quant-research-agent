import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

Z90 = 1.645

NAIVE_COLOR = "#555555"
PALETTE = plt.get_cmap("tab10").colors


# ---------- helpers ----------
def _color(m, models):
    if m == "naive":
        return NAIVE_COLOR
    others = [x for x in models if x != "naive"]
    return PALETTE[others.index(m) % len(PALETTE)]

def _save(fig, out, name):
    p = Path(out) / "figures" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return f"figures/{name}"

def _ratio(f):
    base = f["metrics"]["naive"]["rmse"]
    return {m: r["rmse"] / base for m, r in f["metrics"].items() if m != "naive"}

def _label_bars(ax, bars, texts, fs=7):
    for b, t in zip(bars, texts):
        ax.annotate(t, (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=fs)


# ---------- facts charts: fn(f, out) -> path ----------
def forecast_interval(f, out):
    ms = list(f["forecast"]); fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, m in enumerate(ms):
        r = f["forecast"][m]
        ax.errorbar(i, r["mean"], yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                    fmt="o", capsize=4, color=_color(m, ms))
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms)
    ax.axhline(0, lw=0.8, color="gray")
    ax.set_ylabel(f"{f['h']}-day log return")
    ax.set_title(f"{f['ticker']}: point forecast and 90% interval")
    return _save(fig, out, "forecast_interval.png")

def metrics_bar(f, out, metric="rmse"):
    ms = list(f["metrics"]); vals = [f["metrics"][m][metric] for m in ms]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(ms, vals, color=[_color(m, ms) for m in ms])
    if "naive" in f["metrics"]:
        ax.axhline(f["metrics"]["naive"][metric], ls="--", lw=0.8, color=NAIVE_COLOR)
    _label_bars(ax, bars, [f"{v:.5f}" for v in vals])
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{f['ticker']}: backtest {metric.upper()} (lower is better, naive in grey)")
    return _save(fig, out, f"metrics_{metric}.png")

def rmse_ratio(f, out):
    r = _ratio(f); ms = list(r)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(ms, [r[m] for m in ms],
                  color=["tab:green" if r[m] < 1 else "tab:red" for m in ms])
    ax.axhline(1, color="k", lw=0.8)
    texts = []
    for m in ms:
        p = f["dm"].get(m, {}).get("p_value")
        texts.append(f"{r[m]:.3f}" + (f"\np={p:.3f}" if p is not None else ""))
    _label_bars(ax, bars, texts)
    ax.set_ylabel("RMSE / naive RMSE")
    ax.set_title(f"{f['ticker']}: RMSE ratio vs naive (below 1 = lower error)")
    return _save(fig, out, "rmse_ratio.png")

def dm_pvalues(f, out, alpha=0.05):
    ms = list(f["dm"]); ps = [f["dm"][m]["p_value"] for m in ms]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.bar(ms, ps, color=["tab:green" if p < alpha else "tab:gray" for p in ps])
    ax.axhline(alpha, color="k", ls="--", lw=0.8, label=f"alpha = {alpha}")
    _label_bars(ax, bars, [f"{p:.3f}" for p in ps])
    ax.set_ylim(0, 1.05); ax.set_ylabel("DM p-value vs naive"); ax.legend(fontsize=7)
    ax.set_title(f"{f['ticker']}: Diebold-Mariano test (green = significant)")
    return _save(fig, out, "dm_pvalues.png")

def coverage_width(f, out, target=0.90):
    ms = list(f["metrics"]); has_w = all("width" in f["metrics"][m] for m in ms)
    fig, axes = plt.subplots(1, 2 if has_w else 1, figsize=(10 if has_w else 6, 3.6),
                             squeeze=False)
    ax = axes[0][0]; cols = [_color(m, ms) for m in ms]
    vals = [f["metrics"][m]["cov90"] for m in ms]
    bars = ax.bar(ms, vals, color=cols)
    ax.axhline(target, color="k", ls="--", lw=0.9, label=f"target {target:.2f}")
    n = f.get("n_origins")
    if n:
        band = 1.96 * np.sqrt(target * (1 - target) / n)
        ax.axhspan(target - band, target + band, color="k", alpha=0.1, label="95% sampling band")
    _label_bars(ax, bars, [f"{v:.3f}" for v in vals])
    ax.set_ylim(0, 1.05); ax.set_ylabel("coverage"); ax.legend(fontsize=7, loc="lower right")
    ax.set_title("90% interval coverage")
    if has_w:
        ax2 = axes[0][1]; w = [f["metrics"][m]["width"] for m in ms]
        b2 = ax2.bar(ms, w, color=cols)
        _label_bars(ax2, b2, [f"{v:.4f}" for v in w])
        ax2.set_ylabel("mean interval width (log return)"); ax2.set_title("Interval width")
    fig.suptitle(f["ticker"])
    return _save(fig, out, "coverage_width.png")

def garch_chart(f, out):
    g = f["garch"]; fig, ax = plt.subplots(figsize=(6.5, 3.8))
    path = g.get("sigma_path") or [p["sigma"] for p in g.get("path", [])]
    if path:
        x = np.arange(1, len(path) + 1); ax.plot(x, path, marker="o", label="sigma")
        if g.get("path"):
            ax.plot(x, [p["daily_sigma"] for p in g["path"]], marker="s", ls=":",
                    label="daily sigma")
        ax.set_xlabel("horizon (days)")
    else:
        s = g["sigma"]; lo, hi = g.get("lo", s), g.get("hi", s)
        ax.errorbar(0, s, yerr=[[s - lo], [hi - s]], fmt="o", capsize=5, label="sigma")
        ax.set_xticks([0]); ax.set_xticklabels([f"{f['h']}-day"])
    if g.get("realized") is not None:
        ax.axhline(g["realized"], color="tab:red", ls="--", lw=0.9, label="realized")
    ax.set_ylabel("volatility (log return)"); ax.legend(fontsize=7)
    ax.set_title(f"{f['ticker']}: GARCH(1,1) t-errors volatility")
    return _save(fig, out, "garch.png")


# ---------- predictions charts: fn(df, out, h) -> list[path] ----------
# df columns: ticker, model, origin, target_date, h, last, actual, pred, lower, upper
def load_preds(run_dir):
    return pd.concat([pd.read_csv(p) for p in Path(run_dir).glob("*_preds.csv")],
                     ignore_index=True)

def prepare(preds):
    df = preds.copy()
    for c in ("origin", "target_date"):
        df[c] = pd.to_datetime(df[c])
    df["ret_act"] = np.log(df["actual"] / df["last"])
    df["ret_pred"] = np.log(df["pred"] / df["last"])
    df["sq"] = (df["ret_pred"] - df["ret_act"]) ** 2
    df["inside"] = (df["actual"] >= df["lower"]) & (df["actual"] <= df["upper"])
    return df

def _term(df, h):
    return df[df["h"] == h].sort_values(["ticker", "model", "origin"])

def _tickers(t):
    for tk in sorted(t["ticker"].unique()):
        yield tk, t[t["ticker"] == tk]

def cum_sq_error(df, out, h):
    t = _term(df, h); models = sorted(t["model"].unique()); paths = []
    for tk, g in _tickers(t):
        fig, ax = plt.subplots(figsize=(10, 4))
        for m in models:
            gm = g[g["model"] == m]
            ax.plot(gm["origin"], gm["sq"].cumsum(), label=m, color=_color(m, models))
        ax.set(title=f"{tk}: cumulative squared error (h={h})",
               ylabel="cumulative squared log-return error")
        ax.legend(fontsize=7); paths.append(_save(fig, out, f"cum_sq_error_{tk}.png"))
    return paths

def cum_diff_vs_naive(df, out, h):
    t = _term(df, h); models = sorted(t["model"].unique()); paths = []
    for tk, g in _tickers(t):
        base = g[g["model"] == "naive"].set_index("origin")["sq"].cumsum()
        fig, ax = plt.subplots(figsize=(10, 4))
        for m in models:
            if m == "naive":
                continue
            s = g[g["model"] == m].set_index("origin")["sq"].cumsum()
            ax.plot(s.index, s - base.reindex(s.index), label=m, color=_color(m, models))
        ax.axhline(0, color="k", lw=0.8)
        ax.set(title=f"{tk}: cumulative squared error minus naive (above 0 = worse)",
               ylabel="difference in cumulative squared error")
        ax.legend(fontsize=7); paths.append(_save(fig, out, f"cum_diff_{tk}.png"))
    return paths

def rolling_rmse_ratio(df, out, h, window=60):
    t = _term(df, h); models = sorted(t["model"].unique()); paths = []
    for tk, g in _tickers(t):
        base = g[g["model"] == "naive"].set_index("origin")["sq"].rolling(window).mean()
        fig, ax = plt.subplots(figsize=(10, 4))
        for m in models:
            if m == "naive":
                continue
            s = g[g["model"] == m].set_index("origin")["sq"].rolling(window).mean()
            ax.plot(s.index, np.sqrt(s / base.reindex(s.index)), label=m, color=_color(m, models))
        ax.axhline(1, color="k", lw=0.8)
        ax.set(title=f"{tk}: rolling RMSE ratio vs naive ({window} origins; below 1 = better)",
               ylabel="RMSE / naive RMSE")
        ax.legend(fontsize=7); paths.append(_save(fig, out, f"rolling_ratio_{tk}.png"))
    return paths

def forecast_fan(df, out, h, show=None, origin_idx=-1):
    models = sorted(df["model"].unique())
    show = show or ([m for m in ("naive",) if m in models]
                    + [m for m in models if m != "naive"][:3])
    paths = []
    for tk in sorted(df["ticker"].unique()):
        origins = np.sort(df.loc[df["ticker"] == tk, "origin"].unique())
        o = origins[origin_idx]
        fig, axes = plt.subplots(1, len(show), figsize=(4.5 * len(show), 3.8),
                                 sharey=True, squeeze=False)
        for ax, m in zip(axes[0], show):
            g = df[(df["ticker"] == tk) & (df["model"] == m) & (df["origin"] == o)].sort_values("h")
            if g.empty:
                continue
            last = g["last"].iloc[0]
            x = pd.concat([pd.Series([o]), g["target_date"]]).to_numpy()
            ax.plot(x, np.r_[last, g["actual"]], "ko-", label="actual")
            ax.plot(x, np.r_[last, g["pred"]], "--o", color=_color(m, models), label="forecast")
            ax.fill_between(g["target_date"], g["lower"], g["upper"], alpha=0.25,
                            color=_color(m, models), label="90% interval")
            ax.set_title(m, fontsize=9); ax.tick_params(axis="x", rotation=45)
        axes[0][0].set_ylabel("price"); axes[0][0].legend(fontsize=7)
        fig.suptitle(f"{tk}: forecast fan from origin {pd.Timestamp(o).date()}")
        paths.append(_save(fig, out, f"forecast_fan_{tk}.png"))
    return paths

def coverage_by_year(df, out, h, target=0.90):
    t = _term(df, h)
    cov = (t.assign(year=t["origin"].dt.year)
             .groupby(["year", "model"])["inside"].mean().unstack("model"))
    ax = cov.plot.bar(figsize=(10, 4), color=[_color(m, list(cov.columns)) for m in cov.columns])
    ax.axhline(target, color="k", ls="--", label=f"target {target:.2f}")
    ax.set(title=f"90% interval coverage by origin year (h={h}, pooled tickers)",
           ylabel="coverage", ylim=(0, 1))
    ax.legend(fontsize=7, ncol=2)
    return [_save(ax.figure, out, "coverage_by_year.png")]



# ---------- professional charts: price and return forecasts ----------
def _bdays(last, n):
    return pd.bdate_range(pd.Timestamp(last), periods=n + 1)[1:]

def price_fan(f, out, levels=(0.5, 0.8, 0.9), lookback=60):
    """History plus nested prediction bands (fan chart) and every model's mean path."""
    hs, paths = f["history"], f["paths"]
    ref = "naive" if "naive" in paths else next(iter(paths))
    hd = pd.to_datetime(hs["dates"])[-lookback:]; hc = np.array(hs["close"])[-lookback:]; last = hc[-1]
    p = paths[ref]; n = len(p); x = [hd[-1], *_bdays(hd[-1], n)]
    cum = np.r_[0.0, [d["cum_log_return"] for d in p]]
    sig = np.r_[0.0, [d["hi"] for d in p]] / Z90          # cumulative sigma from the 90% bound
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(hd, hc, color="k", lw=1.1, label="close")
    for lv in sorted(levels, reverse=True):
        z = stats.norm.ppf((1 + lv) / 2)
        ax.fill_between(x, last * np.exp(cum - z * sig), last * np.exp(cum + z * sig),
                        color="tab:blue", alpha=0.18, lw=0, label=f"{lv:.0%} band")
    ms = list(paths)
    for m in ms:
        y = np.r_[last, [d["price"] for d in paths[m]]]
        ax.plot(x, y, marker="o", ms=3, lw=1.3, color=_color(m, ms), label=m)
    ax.axvline(hd[-1], color="gray", ls=":", lw=0.8)
    ax.set_ylabel("price"); ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.set_title(f"{f['ticker']}: price history and {n}-day fan chart (bands around {ref})")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, out, "price_fan.png")

def backtest_band(df, out, h, model=None):
    """Realised h-day return against the forecast and its 90% band; breaches in red."""
    t = _term(df, h); models = sorted(t["model"].unique())
    model = model or ("naive" if "naive" in models else models[0]); paths = []
    for tk, g in _tickers(t[t["model"] == model]):
        lo = np.log(g["lower"] / g["last"]); hi = np.log(g["upper"] / g["last"])
        out_ = ~g["inside"]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(g["target_date"], lo, hi, color="tab:blue", alpha=0.2, label="90% interval")
        ax.plot(g["target_date"], g["ret_pred"], color=_color(model, models), lw=1, label=f"{model} forecast")
        ax.scatter(g.loc[~out_, "target_date"], g.loc[~out_, "ret_act"], s=9, color="k", label="actual")
        ax.scatter(g.loc[out_, "target_date"], g.loc[out_, "ret_act"], s=22, color="tab:red",
                   label=f"outside ({out_.mean():.1%})")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set(title=f"{tk}: realised {h}-day log return vs {model} forecast band",
               ylabel=f"{h}-day log return")
        ax.legend(fontsize=7, ncol=4, loc="upper left")
        paths.append(_save(fig, out, f"backtest_band_{tk}.png"))
    return paths

def pred_vs_actual(df, out, h):
    """Mincer-Zarnowitz: regress actual on forecast. Intercept 0 and slope 1 = unbiased."""
    t = _term(df, h); models = sorted(t["model"].unique())
    ms = [m for m in models if m != "naive"]; paths = []
    for tk, g in _tickers(t):
        fig, axes = plt.subplots(1, len(ms), figsize=(3.8 * len(ms), 3.8), squeeze=False)
        for ax, m in zip(axes[0], ms):
            gm = g[g["model"] == m]; x, y = gm["ret_pred"], gm["ret_act"]
            lim = np.nanmax(np.abs(np.r_[x, y])) * 1.05
            ax.scatter(x, y, s=9, alpha=0.5, color=_color(m, models))
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="45° (perfect)")
            title = m
            if len(gm) > 2 and x.std() > 0:
                r = stats.linregress(x, y); xs = np.array([x.min(), x.max()])
                ax.plot(xs, r.intercept + r.slope * xs, "r-", lw=1.2, label="fit")
                title += f"\nslope={r.slope:.2f}±{1.96*r.stderr:.2f}, a={r.intercept:.4f}, R²={r.rvalue**2:.3f}"
            ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="predicted return", ylabel="actual return")
            ax.set_title(title, fontsize=8); ax.legend(fontsize=6)
        fig.suptitle(f"{tk}: Mincer-Zarnowitz, {h}-day log return")
        paths.append(_save(fig, out, f"pred_vs_actual_{tk}.png"))
    return paths

def error_dist(df, out, h):
    """Histogram with normal fit (top) and QQ plot (bottom) of forecast errors."""
    t = _term(df, h); models = sorted(t["model"].unique()); paths = []
    for tk, g in _tickers(t):
        fig, axes = plt.subplots(2, len(models), figsize=(3.6 * len(models), 6.2), squeeze=False)
        for j, m in enumerate(models):
            e = (g[g["model"] == m]["ret_act"] - g[g["model"] == m]["ret_pred"]).dropna().to_numpy()
            top, bot = axes[0][j], axes[1][j]
            top.hist(e, bins=30, density=True, color=_color(m, models), alpha=0.7)
            xs = np.linspace(e.min(), e.max(), 200)
            top.plot(xs, stats.norm.pdf(xs, e.mean(), e.std()), "k-", lw=1)
            top.set_title(f"{m}\nskew={stats.skew(e):.2f}, ex.kurt={stats.kurtosis(e):.2f}", fontsize=8)
            stats.probplot(e, dist="norm", plot=bot)
            bot.get_lines()[0].set(markersize=3, color=_color(m, models)); bot.set_title("")
            bot.set_xlabel("normal quantiles"); bot.set_ylabel("error quantiles")
        fig.suptitle(f"{tk}: {h}-day forecast error distribution")
        paths.append(_save(fig, out, f"error_dist_{tk}.png"))
    return paths

def _sigma_h(g):
    """h-day sigma implied by the stored 90% interval: ln(U/L) / (2 * 1.645)."""
    return np.log(g["upper"] / g["lower"]) / (2 * Z90)

def pit_hist(df, out, h, bins=10):
    """PIT histogram under a normal with the interval's sigma. Flat = calibrated;
    U = overconfident, hump = underconfident, slope = biased."""
    t = _term(df, h); models = sorted(t["model"].unique()); paths = []
    for tk, g in _tickers(t):
        fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.4),
                                 sharey=True, squeeze=False)
        for ax, m in zip(axes[0], models):
            gm = g[g["model"] == m]
            pit = pd.Series(stats.norm.cdf((gm["ret_act"] - gm["ret_pred"]) / _sigma_h(gm))).dropna()
            n = len(pit); band = 1.96 * np.sqrt((1 / bins) * (1 - 1 / bins) / n)
            ax.hist(pit, bins=bins, range=(0, 1), density=False, weights=np.ones(n) / n,
                    color=_color(m, models), alpha=0.8)
            ax.axhspan(1 / bins - band, 1 / bins + band, color="k", alpha=0.12)
            ax.axhline(1 / bins, color="k", ls="--", lw=0.8)
            ax.set_title(m, fontsize=9); ax.set_xlabel("PIT")
        axes[0][0].set_ylabel("share of origins")
        fig.suptitle(f"{tk}: PIT histogram, {h}-day returns (grey = 95% band under calibration)")
        paths.append(_save(fig, out, f"pit_{tk}.png"))
    return paths


# ---------- professional charts: volatility ----------
def vol_realized(f, out, window=21):
    """Conditional vol vs rolling realised vol, with the GARCH forecast path appended."""
    d = f["garch_diag"]; dates = pd.to_datetime(d["dates"])
    r = pd.Series(d["ret"], index=dates); cv = pd.Series(d["cond_vol"], index=dates)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r.abs().index, r.abs(), ".", ms=2, color="gray", alpha=0.5, label="|daily return|")
    ax.plot(r.rolling(window).std(), color="tab:orange", lw=1.2, label=f"realised vol ({window}d)")
    ax.plot(cv, color="tab:blue", lw=1.2, label=f"conditional vol ({d['model']})")
    path = f["garch"].get("path")
    if path:
        x = _bdays(dates[-1], len(path))
        ax.plot(x, [p["daily_sigma"] for p in path], color="tab:red", marker="o", ms=3,
                label="forecast daily sigma")
    ax.set(ylabel="daily volatility (log return)",
           title=f"{f['ticker']}: conditional vs realised volatility")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    return _save(fig, out, "vol_realized.png")

def _acf(x, nlags=20):
    x = np.asarray(x, float) - np.mean(x); d = x @ x
    return np.array([x[:-k] @ x[k:] / d for k in range(1, nlags + 1)])

def garch_diag(f, out):
    """2x2: standardised residuals, QQ vs fitted t, ACF of residuals and of squares."""
    d = f["garch_diag"]; z = np.asarray(d["std_resid"], float); n = len(z)
    nu = d.get("nu"); lags = np.arange(1, 21); band = 1.96 / np.sqrt(n)
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.5))
    ax[0][0].plot(pd.to_datetime(d["dates"]), z, lw=0.6)
    ax[0][0].set_title("standardised residuals"); ax[0][0].tick_params(axis="x", rotation=30)
    dist = stats.t(nu, scale=np.sqrt((nu - 2) / nu)) if nu and nu > 2 else stats.norm
    q = (np.arange(1, n + 1) - 0.5) / n
    ax[0][1].plot(dist.ppf(q), np.sort(z), ".", ms=3)
    lim = [dist.ppf(q[0]), dist.ppf(q[-1])]; ax[0][1].plot(lim, lim, "r-", lw=0.8)
    ax[0][1].set(title=f"QQ vs {'t(nu=%.1f)' % nu if nu else 'normal'}",
                 xlabel="theoretical", ylabel="sample")
    for a, series, ttl in ((ax[1][0], z, "ACF of residuals"), (ax[1][1], z ** 2, "ACF of squared residuals")):
        a.bar(lags, _acf(series), width=0.6); a.axhline(band, color="r", ls="--", lw=0.8)
        a.axhline(-band, color="r", ls="--", lw=0.8); a.set(title=ttl, xlabel="lag")
    fig.suptitle(f"{f['ticker']}: {d['model']} residual diagnostics (bars inside red lines = no leftover structure)")
    return _save(fig, out, "garch_diag.png")

def news_impact(f, out):
    """News impact curve: next-day variance as a function of today's shock (GARCH/GJR)."""
    d = f["garch_diag"]; p = d["params"]
    if d.get("vol") != "GARCH":
        raise ValueError("news impact curve implemented for GARCH/GJR only")
    om, al, be, ga = p["omega"], p["alpha[1]"], p["beta[1]"], p.get("gamma[1]", 0.0)
    pers = al + be + ga / 2; lr = om / (1 - pers)
    sd = np.sqrt(lr); e = np.linspace(-5 * sd, 5 * sd, 201)
    var = om + (al + ga * (e < 0)) * e ** 2 + be * lr
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(e / 100, np.sqrt(var) / 100, color="tab:blue")
    ax.axvline(0, color="gray", lw=0.6)
    ax.set(xlabel="shock (daily log return)", ylabel="next-day sigma",
           title=f"{f['ticker']}: news impact curve (persistence {pers:.3f})")
    return _save(fig, out, "news_impact.png")

def vol_vs_realized(df, out, h):
    """Interval-implied h-day sigma against the realised |h-day return|."""
    t = _term(df, h); models = sorted(t["model"].unique())
    model = "naive" if "naive" in models else models[0]; paths = []
    for tk, g in _tickers(t[t["model"] == model]):
        s = _sigma_h(g); k = np.sqrt(2 / np.pi)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(g["origin"], s, color="tab:blue", label="forecast sigma (h-day)")
        ax.plot(g["origin"], k * s, color="tab:blue", ls="--", lw=0.9, label="expected |return| = 0.80 sigma")
        ax.scatter(g["origin"], g["ret_act"].abs(), s=9, color="k", label="realised |return|")
        ax.set(title=f"{tk}: forecast volatility vs realised absolute {h}-day return",
               ylabel="log return")
        ax.legend(fontsize=7, ncol=3, loc="upper left")
        paths.append(_save(fig, out, f"vol_vs_realized_{tk}.png"))
    return paths

# ---------- registry ----------
FACTS_CHARTS = dict(forecast=forecast_interval, price_fan=price_fan, rmse=metrics_bar,
                    ratio=rmse_ratio, dm=dm_pvalues, coverage=coverage_width, garch=garch_chart,
                    vol_hist=vol_realized, garch_diag=garch_diag, nic=news_impact)
PRED_CHARTS = dict(band=backtest_band, scatter=pred_vs_actual, err_dist=error_dist, pit=pit_hist,
                   vol_real=vol_vs_realized, cum_err=cum_sq_error, cum_diff=cum_diff_vs_naive,
                   rolling=rolling_rmse_ratio, fan=forecast_fan, cov_year=coverage_by_year)

def build_all(f, out, preds=None, h=None):
    """Return {key: path | [paths] | 'MISSING (...)'}. A failed chart never stops the report."""
    h = h or f.get("h", 5); figs = {}
    def run(key, fn, *a):
        try:
            figs[key] = fn(*a)
        except Exception as e:
            plt.close("all"); figs[key] = f"MISSING ({type(e).__name__}: {e})"
    for key, fn in FACTS_CHARTS.items():
        run(key, fn, f, out)
    if preds is not None:
        df = prepare(preds)
        for key, fn in PRED_CHARTS.items():
            run(key, fn, df, out, h)
    return figs
