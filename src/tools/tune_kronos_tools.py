# src/tools/tune_kronos_tools.py
import hashlib
import itertools
import json
import time
from datetime import datetime, timezone

import pandas as pd

from evaluation.walk_forward import walk_forward
from evaluation.metrics import score
from models.kronos_model import get_forecaster
from .config import KRONOS_SAMPLES, MIN_HIST, make_vol
from .cache import get_dataset, clean_json, STORE
from .errors import ToolError
from .forecast_tools import make_model
from .eval_tools import CODE_VERSION
from .registry_core import tool
from .schemas import TuneKronosArgs

DEFAULT_GRID = dict(lookback=[60, 120, 180, 250], T=[0.3, 0.6, 1.0])
MIN_VAL_ORIGINS = 30
MIN_COVERAGE = 0.6
TUNED_PATH = STORE / "tuned_config.json"

def _cache_path(df_val, dataset_id, size, cfg, h, start, step):
    raw = json.dumps([CODE_VERSION, "tune", dataset_id, size,
                      str(df_val["timestamps"].iloc[-1]), cfg, h, start, step,
                      KRONOS_SAMPLES], sort_keys=True, default=str)
    d = STORE / "tune_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha1(raw.encode()).hexdigest()[:12] + ".pkl")


def _score_row(res, name, dataset_id):
    return score(res.assign(model=name, ticker=dataset_id)).reset_index().iloc[0].to_dict()


def _md(tab):
    cols = ["lookback", "T", "MAE", "RMSE", "dir_acc", "coverage", "width", "beats_naive"]
    lines = ["| " + " | ".join(cols) + " |", "|:--|" + "--:|" * (len(cols) - 1)]
    for _, r in tab.iterrows():
        cells = [str(r["lookback"]), str(r["T"])]
        cells += [f"{r[c]:.4f}" for c in cols[2:-1]] + [str(bool(r["beats_naive"]))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


@tool(TuneKronosArgs, slow=True)
def tune_kronos(args) -> dict:
    """Pick Kronos lookback and temperature on a validation zone, never touching the test zone.

    Returns a ranked table, the chosen config (or 'use_naive'), and test_start for the
    final run_backtest. Do not re-tune after seeing test results.
    """
    g = lambda k, d: getattr(args, k, None) if getattr(args, k, None) is not None else d
    dataset_id = args.dataset_id
    size, h, step = g("size", "small"), g("h", 5), g("step", 5)
    val_origins, test_origins = g("val_origins", MIN_VAL_ORIGINS), g("test_origins", 12)
    grid = g("grid", DEFAULT_GRID)

    if val_origins < MIN_VAL_ORIGINS:
        raise ToolError("TOO_FEW_ORIGINS", f"val_origins={val_origins} is too few.",
                        f"Use at least {MIN_VAL_ORIGINS}.")

    df = get_dataset(dataset_id)
    test_rows = test_origins * step + h
    val_rows = val_origins * step + h
    if len(df) - test_rows - val_rows < MIN_HIST:
        raise ToolError("NOT_ENOUGH_DATA", "Not enough history for warm-up, validation and test.",
                        "Use an earlier data start or fewer origins.")

    df_val = df.iloc[:len(df) - test_rows].reset_index(drop=True)   # test zone removed
    start = -val_rows
    vol = make_vol(cached=True)
    vol.reset()

    naive = walk_forward(df_val, make_model("naive"), horizon=h, start=start, step=step,
                         vol_model=vol, terminal_only=True, force_intervals=True)
    n_val = int(naive["origin"].nunique())
    if n_val < MIN_VAL_ORIGINS:
        raise ToolError("TOO_FEW_ORIGINS", f"Only {n_val} validation origins.",
                        "Increase val_origins or reduce step.")
    naive_mae = float(_score_row(naive, "naive", dataset_id)["MAE"])

    rows = []
    for lb, T in itertools.product(grid["lookback"], grid["T"]):
        cfg = dict(lookback=lb, T=T)
        p = _cache_path(df_val, dataset_id, size, cfg, h, start, step)
        t0 = time.time()
        if p.exists():
            res = pd.read_pickle(p)
        else:
            m = get_forecaster(size=size, n_samples=KRONOS_SAMPLES, lookback=lb, T=T)
            res = walk_forward(df_val, m, horizon=h, start=start, step=step,
                               vol_model=vol, terminal_only=True, force_intervals=True)
            tmp = p.with_suffix(".tmp")
            res.to_pickle(tmp)
            tmp.replace(p)
        s = _score_row(res, f"kronos_{size}", dataset_id)
        rows.append(dict(lookback=lb, T=T, MAE=s["MAE"], RMSE=s["RMSE"],
                         dir_acc=s["dir_acc"], coverage=s["coverage"], width=s["width"],
                         secs=round(time.time() - t0, 1)))

    tab = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    tab["beats_naive"] = (tab["MAE"] < naive_mae) & (tab["coverage"] >= MIN_COVERAGE)

    # stability: a neighbouring lookback (same T) must also pass
    lbs = sorted(grid["lookback"])
    passed = {(r.lookback, r.T) for r in tab.itertuples() if r.beats_naive}

    def stable(lb, T):
        i = lbs.index(lb)
        nb = [lbs[j] for j in (i - 1, i + 1) if 0 <= j < len(lbs)]
        return any((n, T) in passed for n in nb)

    chosen = next(({"lookback": int(r.lookback), "T": float(r.T)}
                   for r in tab.itertuples() if r.beats_naive and stable(r.lookback, r.T)),
                  None)

    out = dict(dataset_id=dataset_id, size=size, h=h, step=step,
               chosen=chosen or "use_naive", naive_val_mae=naive_mae,
               n_val_origins=n_val, n_configs=len(tab), grid=grid,
               test_start=-test_rows,
               tuned_at=datetime.now(timezone.utc).isoformat())
    if chosen:  # only persist a real choice
        TUNED_PATH.write_text(json.dumps(out, default=str))
    return clean_json(dict(ok=True, table=_md(tab), **out))
