"""Phase 2 entry point: baselines vs Kronos variants.

Workflow:
    1. python scripts/run_kronos_wf.py            # slow, once per variant
    2. python scripts/run_phase2.py --step 5 --start -500 \
           --kronos-saved kronos_small_lb30 kronos_small_lb60 \
                          kronos_small_lb90 kronos_small_ft_lb90

Other usage:
    python scripts/run_phase2.py                          # baselines only, step=1
    python scripts/run_phase2.py --kronos small           # run Kronos live
    python scripts/run_phase2.py ... --benchmarks naive kronos_small_lb90
    python scripts/run_phase2.py ... --no-cache           # refit baselines

Output: results/phase2/<timestamp>/
    <ticker>_preds.csv, accuracy.csv, accuracy_all_h.csv, dm.csv,
    dm_vs_<benchmark>.csv (extra benchmarks), volatility.csv,
    findings_draft.md, config.json

Visualization lives in notebooks/, which load these CSVs.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.loader import fetch_ohlcv  # noqa: E402
from models.baselines import Naive, Drift, Arima  # noqa: E402
from models.volatility import VolModel, qlike, CachedVol, Z90  # noqa: E402
from evaluation.walk_forward import walk_forward  # noqa: E402
from evaluation.metrics import score, add_returns  # noqa: E402
from evaluation.dm_test import dm_table  # noqa: E402

# ---- Fixed settings (identical for every model) ----
H, MIN_HIST, SEED, N_SAMPLES = 5, 400, 42, 10
PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "arch", "torch"]
RESULTS = ROOT / "results"
CACHE = RESULTS / "phase2" / "_cache"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["MU", "KO"])
    p.add_argument("--start", default="2023-01-01",
                   help="date, or int row offset such as -500")
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--force-intervals", action="store_true",
                   help="GARCH bands for ALL models")
    p.add_argument("--kronos", nargs="+", default=[],
                   choices=["mini", "small", "base", "small_ft"],
                   help="run these Kronos variants live (slow)")
    p.add_argument("--lookbacks", nargs="+", type=int, default=[90],
                   help="lookbacks for live Kronos runs")
    p.add_argument("--kronos-saved", nargs="+", default=[],
                   help="folder names under results/, e.g. kronos_small_lb90")
    p.add_argument("--benchmarks", nargs="+", default=["naive"],
                   help="model names used as DM benchmarks (first is the main one)")
    p.add_argument("--no-cache", action="store_true", help="refit baselines")
    a = p.parse_args()
    if a.start.lstrip("-").isdigit():
        a.start = int(a.start)
    return a


# ------------------------------------------------------------------ backtest
def run_all(data, models, args, vol):
    """Ticker-outer loop so the GARCH cache is valid and cleared per ticker."""
    parts = {m.name: [] for m in models}
    for tk, df in data.items():
        vol.reset()
        for m in models:
            print(f"[{tk}] {m.name} ...", flush=True)
            r = walk_forward(df, m, horizon=H, start=args.start, step=args.step,
                             min_hist=MIN_HIST, vol_model=vol,
                             force_intervals=args.force_intervals)
            parts[m.name].append(r.assign(ticker=tk, model=m.name))
    return {n: pd.concat(p, ignore_index=True) for n, p in parts.items()}


def cached_baselines(data, models, args, vol):
    """Run baselines, or load them if settings and data are unchanged."""
    payload = dict(
        tickers=args.tickers, start=args.start, step=args.step, H=H,
        min_hist=MIN_HIST, force=args.force_intervals, vol=vol.config,
        models=[m.config for m in models],
        last={tk: str(df["timestamps"].iloc[-1]) for tk, df in data.items()},
    )
    key = hashlib.md5(json.dumps(payload, sort_keys=True,
                                 default=str).encode()).hexdigest()[:12]
    f = CACHE / f"baselines_{key}.pkl"
    if f.exists() and not args.no_cache:
        print(f"Loaded cached baselines ({f.name})")
        return pd.read_pickle(f)
    res = run_all(data, models, args, vol)
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(res, f)
    return res


# ------------------------------------------------------------------ saved runs
def check_settings(run_dir, name, args):
    """Stop if a saved run used different settings from this one."""
    cfg_file = run_dir / "config.json"
    if not cfg_file.exists():
        print(f"  warning: {name} has no config.json, settings not verified")
        return
    cfg = json.loads(cfg_file.read_text())
    bad = [f"{k}: saved={cfg.get(k)!r}, now={v!r}"
           for k, v in dict(horizon=H, start=args.start, step=args.step).items()
           if cfg.get(k) != v]
    missing = set(args.tickers) - set(cfg.get("tickers", args.tickers))
    if missing:
        bad.append(f"tickers missing in saved run: {sorted(missing)}")
    if bad:
        raise ValueError(f"{name} settings differ from this run:\n  " + "\n  ".join(bad))


def load_saved(name, args):
    """Latest predictions.csv under results/<name>/<date>/."""
    base = RESULTS / name
    if not base.is_dir():
        raise FileNotFoundError(f"results/{name} not found. Folders: "
                                f"{sorted(p.name for p in RESULTS.iterdir() if p.is_dir())}")
    runs = sorted(p for p in base.iterdir() if (p / "predictions.csv").exists())
    if not runs:
        raise FileNotFoundError(f"No predictions.csv under {base}")
    run = runs[-1]
    check_settings(run, name, args)
    df = pd.read_csv(run / "predictions.csv", parse_dates=["origin", "target_date"])
    if "ticker" not in df.columns:
        raise ValueError(f"{name}: no ticker column, rerun run_kronos_wf.py")
    df = df[df["ticker"].isin(args.tickers)].assign(model=name)
    print(f"Loaded {name} from {run.name} ({len(df)} rows)")
    return df


def align(results):
    """Keep only (ticker, origin, h) rows present for every model."""
    keys = None
    for r in results.values():
        k = pd.MultiIndex.from_frame(r[["ticker", "origin", "h"]])
        keys = k if keys is None else keys.intersection(k)
    if len(keys) == 0:
        raise ValueError("No common (ticker, origin, h) rows. Check start, step and "
                         "data snapshot of every run.")
    print(f"Common (ticker, origin, h) rows: {len(keys)}")
    out = {}
    for n, r in results.items():
        mask = pd.MultiIndex.from_frame(r[["ticker", "origin", "h"]]).isin(keys)
        if (~mask).any():
            print(f"  {n}: dropped {int((~mask).sum())} non-common rows")
        out[n] = r[mask].reset_index(drop=True)
    return out


# ------------------------------------------------------------------ tables
def accuracy_tables(allres):
    full = score(allres)                                   # ticker, model, h
    t1 = (full.xs(H, level="h")
              .rename(columns={"dir_acc": "DirAcc", "coverage": "Cov90",
                               "width": "Width"}))
    return full, t1


def vol_eval(results, data):
    """Table 3 without refitting GARCH.

    Naive's price bands are last * exp(+-z * sigma), so
    sigma_hat = log(upper / last) / z. Realized sigma comes from a cumulative
    sum of squared log returns.
    """
    base = results["naive"]
    base = base.loc[base["h"] == H]
    rows = []
    for tk, g in base.groupby("ticker"):
        df = data[tk]
        px = df["close"].to_numpy(float)
        csum = np.concatenate([[0.0], np.cumsum(np.diff(np.log(px)) ** 2)])
        pos = pd.DatetimeIndex(pd.to_datetime(df["timestamps"])).get_indexer(g["origin"])
        if (pos < 0).any():
            raise ValueError("Origin not found in price data.")
        s_real = np.sqrt(csum[pos + H] - csum[pos])
        s_hat = np.log(g["upper"] / g["last"]).to_numpy() / Z90
        rows.append(dict(ticker=tk, n=len(g),
                         QLIKE=qlike(s_hat, s_real),
                         sigma_RMSE=float(np.sqrt(np.mean((s_hat - s_real) ** 2))),
                         mean_sigma_pred=float(s_hat.mean()),
                         mean_sigma_real=float(s_real.mean())))
    return pd.DataFrame(rows).set_index("ticker")


# ------------------------------------------------------------------ findings
def write_findings(t1, dm, vol, args, path):
    L = ["# Findings (auto-generated draft: verify and edit before use)", ""]
    L.append(f"- Setup: h={H}, step={args.step}, start={args.start}, "
             f"{int(t1['n'].min())}-{int(t1['n'].max())} origins per ticker and model.")
    sig = dm[dm["verdict"].isin(["A better", "A worse"])]
    if sig.empty:
        L.append(f"- Nothing significantly differs from Naive at h={H} "
                 f"(smallest p = {dm['p'].min():.3f}).")
    else:
        L.append("- Significant vs Naive (p<0.05): " + "; ".join(
            f"{t}/{m} {r.verdict} (DM={r.dm:.2f}, p={r.p:.3f})"
            for (t, m), r in sig.iterrows()) + ".")
    for tk, g in t1.groupby(level="ticker"):
        best = g["RMSE"].idxmin()[1]
        nv = g["RMSE"].get((tk, "naive"), np.nan)
        L.append(f"- {tk}: lowest RMSE is {best} ({g['RMSE'].min():.4f}); "
                 f"Naive RMSE {nv:.4f}.")
    nn = t1[t1.index.get_level_values("model") != "naive"]
    L.append("- DirAcc (vs 0.50): " + "; ".join(
        f"{t}/{m} {r.DirAcc:.3f} (n={int(r.n_direction)})"
        for (t, m), r in nn.iterrows()) + ".")
    cv = t1[t1["n_intervals"] > 0]
    L.append("- Cov90 (target 0.90) and width: " + "; ".join(
        f"{t}/{m} {r.Cov90:.3f}"
        + (" [narrow]" if r.Cov90 < 0.85 else " [wide]" if r.Cov90 > 0.95 else "")
        + f", width {r.Width:.4f}" for (t, m), r in cv.iterrows()) + ".")
    L.append("- GARCH QLIKE vs realized vol: " + "; ".join(
        f"{t} {r.QLIKE:.4f} (mean sigma pred {r.mean_sigma_pred:.4f} vs real "
        f"{r.mean_sigma_real:.4f})" for t, r in vol.iterrows()) + ".")
    path.write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ config
def versions():
    v = {}
    for p in PACKAGES:
        try:
            v[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return v


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


# ------------------------------------------------------------------ main
def build_models(args):
    """Return (baselines, live Kronos models)."""
    base = [Naive(), Drift(), Arima()]
    live = []
    if args.kronos:
        from models.kronos_model import get_forecaster
        for size in args.kronos:
            for lb in args.lookbacks:
                live.append(get_forecaster(size, n_samples=N_SAMPLES,
                                           lookback=lb, seed=SEED))
    names = [m.name for m in base + live] + list(args.kronos_saved)
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"Duplicate model names: {dup}")
    return base, live


def main():
    args = parse_args()
    np.random.seed(SEED)
    out = RESULTS / "phase2" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    data = {tk: fetch_ohlcv(tk) for tk in args.tickers}   # cache only
    vol = CachedVol("GARCH", 1, 0, 1, "t")                # frozen before test results
    base, live = build_models(args)

    results = cached_baselines(data, base, args, vol)
    if live:
        results.update(run_all(data, live, args, vol))
    for name in args.kronos_saved:
        results[name] = load_saved(name, args)
    results = align(results)

    main_bench = args.benchmarks[0]
    for b in args.benchmarks:
        if b not in results:
            raise ValueError(f"Benchmark {b!r} not in models: {sorted(results)}")

    allres = pd.concat(results.values(), ignore_index=True)
    ret = add_returns(allres)
    for tk, g in ret.groupby("ticker"):                   # {ticker}_preds.csv
        g.to_csv(out / f"{tk}_preds.csv", index=False)

    full, t1 = accuracy_tables(allres)
    full.to_csv(out / "accuracy_all_h.csv")
    t1.to_csv(out / "accuracy.csv")

    dm = dm_table(results, benchmark=main_bench, h=H, step=args.step, loss="squared")
    dm.to_csv(out / "dm.csv")
    for b in args.benchmarks[1:]:
        dm_table(results, benchmark=b, h=H, step=args.step,
                 loss="squared").to_csv(out / f"dm_vs_{b}.csv")

    volt = vol_eval(results, data)
    volt.to_csv(out / "volatility.csv")
    write_findings(t1, dm, volt, args, out / "findings_draft.md")

    origins = allres.groupby("ticker")["origin"].agg(["min", "max", "nunique"])
    config = dict(
        created=datetime.now().isoformat(timespec="seconds"),
        tickers=args.tickers, horizon=H, start=args.start, step=args.step,
        min_hist=MIN_HIST, seed=SEED, n_samples=N_SAMPLES,
        force_intervals=args.force_intervals, vol_model=vol.config,
        benchmarks=args.benchmarks,
        evaluation=dict(target="cumulative log return from origin close",
                        dm=dict(loss="squared", step=args.step, benchmark=main_bench)),
        models=[m.config for m in base + live],
        saved_runs=args.kronos_saved,
        origins={tk: dict(first=str(r["min"].date()), last=str(r["max"].date()),
                          n=int(r["nunique"])) for tk, r in origins.iterrows()},
        data={tk: dict(n_rows=len(df),
                       first_date=str(pd.Timestamp(df["timestamps"].iloc[0]).date()),
                       last_date=str(pd.Timestamp(df["timestamps"].iloc[-1]).date()))
              for tk, df in data.items()},
        python=platform.python_version(), packages=versions(), git_commit=git_commit(),
    )
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)
    print(f"\n== Table 1: accuracy (h={H}) ==\n", t1)
    print(f"\n== Table 2: DM vs {main_bench} ==\n", dm)
    print("\n== Table 3: volatility ==\n", volt)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()