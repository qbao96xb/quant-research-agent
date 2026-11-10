from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
HORIZON = 5
KEYS = ["ticker", "origin", "h"]


def latest_csv(model_dir):
    runs = sorted(p for p in model_dir.iterdir() if (p / "predictions.csv").exists())
    return runs[-1] / "predictions.csv" if runs else None


preds = {}
for d in sorted(RES.iterdir()):
    if d.is_dir():
        f = latest_csv(d)
        if f is not None:
            df = pd.read_csv(f, parse_dates=["origin", "target_date"])
            if "ticker" not in df.columns:
                print(f"skip {d.name}: no ticker column (old run, delete it)")
                continue
            preds[d.name] = df

if "naive" not in preds:
    raise SystemExit("Run run_baselines.py first (naive is required).")

# ---- keep only rows common to all models ----
common = None
for p in preds.values():
    k = set(zip(p["ticker"], p["origin"], p["h"]))
    common = k if common is None else common & k
print(f"Common (ticker, origin, h) rows: {len(common)}")
for name, p in preds.items():
    mask = [k in common for k in zip(p["ticker"], p["origin"], p["h"])]
    dropped = len(p) - sum(mask)
    if dropped:
        print(f"  {name}: dropped {dropped} non-common rows")
    preds[name] = p[mask].reset_index(drop=True)


def dm_test(e1, e2, h):
    """DM test, squared-error loss, Harvey small-sample correction. d>0 means model 1 is worse."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    if n < 3:
        return np.nan, np.nan
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))


naive = preds["naive"]


def summarize(name, p, naive_p, label):
    p = p.assign(
        err=p["pred"] - p["actual"],
        dir_ok=np.sign(p["pred"] - p["last"]) == np.sign(p["actual"] - p["last"]),
    )
    has_int = p["lower"].notna().all()
    cover = ((p["actual"] >= p["lower"]) & (p["actual"] <= p["upper"])).mean() if has_int else np.nan

    # DM at h=5; origins are non-overlapping (step=5), so lag truncation h=1
    m = p.merge(naive_p[KEYS + ["actual", "pred"]], on=KEYS, suffixes=("", "_naive"))
    m = m[m["h"] == HORIZON].sort_values(["ticker", "origin"])
    if name == "naive" or len(m) < 10:
        stat, pval = np.nan, np.nan
    else:
        stat, pval = dm_test((m["pred"] - m["actual"]).to_numpy(),
                             (m["pred_naive"] - m["actual"]).to_numpy(), h=1)

    return dict(
        model=name,
        scope=label,
        n_origins=p.groupby("ticker")["origin"].nunique().sum(),
        MAE=p["err"].abs().mean(),
        RMSE=np.sqrt((p["err"] ** 2).mean()),
        DirAcc=np.nan if name == "naive" else p["dir_ok"].mean(),
        Coverage90=cover,
        DM_stat=stat,
        DM_p=pval,
    )


# ---- pooled table ----
rows = [summarize(n, p, naive, "ALL") for n, p in preds.items()]
table = pd.DataFrame(rows).set_index("model").round(4)
print("\n=== Pooled (MU + KO) ===")
print(table.drop(columns="scope").to_string())

# ---- per-ticker tables ----
per_ticker = []
for t in sorted(naive["ticker"].unique()):
    rows_t = [summarize(n, p[p["ticker"] == t], naive[naive["ticker"] == t], t)
              for n, p in preds.items()]
    tt = pd.DataFrame(rows_t).set_index("model").round(4)
    print(f"\n=== {t} ===")
    print(tt.drop(columns="scope").to_string())
    per_ticker.append(tt)

pd.concat([table] + per_ticker).to_csv(RES / "comparison.csv")

# ---- per-horizon MAE ----
print("\nPer-horizon MAE (pooled):")
print(pd.DataFrame({n: p.assign(ae=(p["pred"] - p["actual"]).abs()).groupby("h")["ae"].mean()
                    for n, p in preds.items()}).round(4).to_string())

# ---- level bias at each horizon ----
print("\nMean (pred - last) by horizon:")
print(pd.DataFrame({n: (p["pred"] - p["last"]).groupby(p["h"]).mean()
                    for n, p in preds.items()}).round(2).to_string())
