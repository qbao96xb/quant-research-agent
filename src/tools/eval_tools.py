# src/tools/eval_tools.py
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd

from evaluation.walk_forward import walk_forward
from evaluation.metrics import score
from evaluation.dm_test import dm_table
from evaluation.cw_test import cw_table
from .config import KRONOS_SAMPLES, MIN_HIST, VOL_SPEC, make_vol
from .kronos_cfg import get_kronos_cfg
from .cache import put_as, get, get_dataset, get_meta, clean_json, data_fp, STORE
from .errors import ToolError
from .forecast_tools import make_model
from .validation import check_horizon
from .estimate import estimate_s, update_calib, n_origins, fits_per_model
from .registry_core import tool
from .schemas import BacktestArgs, MetricsArgs, DMArgs, CompareArgs, StatusArgs

# bump when model or volatility code changes (invalidates the cache)
CODE_VERSION = "v9"   # v9: data fingerprint in cache keys, consistency guard
SYNC_LIMIT = 100      # seconds; run inline below this, in the background above it
IGNORE_CFG = {"device", "n_jobs"}   # settings that must not change the cache key


# ---------- helpers ----------
def _model_cache_path(fp, dataset_id, name, h, start, step, cfg):
    cfg = {k: v for k, v in cfg.items() if k not in IGNORE_CFG}
    raw = json.dumps([CODE_VERSION, dataset_id, fp,
                      name, h, str(start), step, cfg, VOL_SPEC, True, MIN_HIST],
                     default=str, sort_keys=True)
    d = STORE / "bt_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha1(raw.encode()).hexdigest()[:12] + ".pkl")

def _start(v):
    """Row offset ("-500" or -500) becomes int. A date stays a string."""
    return int(v) if str(v).lstrip("-").isdigit() else v


def _is_kronos(name):
    return name.startswith("kronos_")

def _config(name, order=None):
    """Model config for the cache key. Kronos is built without loading weights."""
    if _is_kronos(name):
        return dict(model=name, n_samples=KRONOS_SAMPLES, **get_kronos_cfg())
    cfg = dict(make_model(name, order).config)
    if name == "arima":
        cfg["order"] = list(order) if order else [1, 0, 1]   # explicit, in case .config omits it
    return cfg

def backtest_key(dataset_id, names, h, start, step, fp, order=None):
    """Same arguments, config and data -> same backtest_id."""
    parts = [CODE_VERSION, dataset_id, fp, sorted(names), h, str(start), step,
             get_kronos_cfg()]
    if order:                                  # appended only when set, so old IDs stay valid
        parts.append(list(order))
    raw = json.dumps(parts, sort_keys=True)
    return "bt_" + hashlib.sha1(raw.encode()).hexdigest()[:10]


def _put_meta(bid, meta):
    p = STORE / f"{bid}.meta.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta))
    tmp.replace(p)

def _get_meta(bid):
    p = STORE / f"{bid}.meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _check_model(bt, model):
    if model not in set(bt["model"]):
        raise ToolError("BAD_MODEL", f"{model!r} is not in this backtest",
                        f"Models here: {sorted(bt['model'].unique())}")

def _check_consistent(frames):
    """Every model must share the same origins, last and actual."""
    ref = frames[0]
    keys = [c for c in ("origin", "h") if c in ref.columns]
    cols = [c for c in ("last", "actual") if c in ref.columns]
    ref = ref.sort_values(keys).reset_index(drop=True)
    for f in frames[1:]:
        g = f.sort_values(keys).reset_index(drop=True)
        same = (len(ref) == len(g)
                and ref[keys].equals(g[keys])
                and all(np.array_equal(ref[c].to_numpy(), g[c].to_numpy(), equal_nan=True)
                        for c in cols))
        if not same:
            raise ToolError("INCONSISTENT_DATA",
                            "Models were backtested on different data.",
                            "Delete data/cache/artifacts/bt_cache and rerun.")

def _load_bt(bid):
    """Load a backtest, or raise a clear error if it is running or failed."""
    from backtest.jobs import read_state
    s = read_state(bid)
    if s and s["status"] == "running":
        raise ToolError("NOT_READY", "Backtest still running.", "Call backtest_status.")
    if s and s["status"] == "failed":
        raise ToolError("BACKTEST_FAILED", s.get("error", ""), "Rerun with a larger step.")
    return get(bid, "bt")


# ---------- backtest ----------
def run_backtest_blocking(bid, args, on_model_done=None):
    """The actual backtest loop. args is a plain dict (it crosses the process boundary)."""
    df = get_dataset(args["dataset_id"])
    fp = data_fp(df)
    start = _start(args["start"])
    names = args["models"]
    order = args.get("order")  # list or None (plain dict crosses processes)
    n = n_origins(df, start, args["h"], args["step"])
    vol = make_vol(cached=True)
    vol.reset()

    frames = []
    for i, name in enumerate(names, 1):
        o = order if name == "arima" else None
        p = _model_cache_path(fp, args["dataset_id"], name, args["h"],
                              args["start"], args["step"], _config(name, o))
        if p.exists():
            res = pd.read_pickle(p)
        else:
            m = make_model(name, o)                 # build only on a cache miss
            t0 = time.time()
            res = walk_forward(df, m, horizon=args["h"], start=start,
                               step=args["step"], vol_model=vol, terminal_only=True,
                               force_intervals=True)

            if res.empty:
                raise ToolError("NOT_ENOUGH_DATA", f"No origins at or after {args['start']}",
                                "Use an earlier start date.")
            update_calib(name, time.time() - t0, fits_per_model(name, n))
            tmp = p.with_suffix(".tmp")
            res.to_pickle(tmp)
            tmp.replace(p)
        frames.append(res.assign(model=name, ticker=args["dataset_id"]))
        if on_model_done:
            on_model_done(i)

    _check_consistent(frames)
    bt = pd.concat(frames, ignore_index=True)
    put_as(bid, "bt", bt, args)
    _put_meta(bid, dict(step=args["step"], h=args["h"], models=names,
                        start=args["start"], ds=args["dataset_id"], order=order))

    return bt

@tool(BacktestArgs, slow=True)
def run_backtest(args) -> dict:
    """Run a walk-forward backtest for several models. Naive is always added automatically.

    Returns a backtest_id. If status is 'running', call backtest_status until done.
    Then call compare_models with that ID.
    Kronos models are slow: use start about -100 and a step of 5 or more.
    A step smaller than h is raised to h automatically.
    """
    from backtest.jobs import start_job, read_state

    df = get_dataset(args.dataset_id)
    fp = data_fp(df)
    check_horizon(args.h, len(df))

    # clamp step to h: faster and non-overlapping targets
    note = None
    step = args.step
    if step < args.h:
        note = f"step raised from {step} to {args.h} (faster, non-overlapping)"
        step = args.h

    def _out(d):
        if note:
            d["note"] = note
        return clean_json(d)

    start = _start(args.start)
    names = list(dict.fromkeys(["naive", *args.models]))
    order = list(args.order) if args.order else None
    bid = backtest_key(args.dataset_id, names, args.h, args.start, step, fp, order)
    job = dict(dataset_id=args.dataset_id, models=names, h=args.h,
               start=args.start, step=step, order=order)

    s = read_state(bid)
    if s and s["status"] in ("running", "done"):  # same call repeated
        return _out(dict(ok=True, backtest_id=bid, status=s["status"], order=order))

    paths = {n: _model_cache_path(fp, args.dataset_id, n, args.h, args.start, step,
                                  _config(n, order if n == "arima" else None))
             for n in names}
    todo = [n for n in names if not paths[n].exists()]
    est = 0
    if todo:
        est = (estimate_s(df, todo, start, args.h, step)
               + 0.015 * n_origins(df, start, args.h, step))
        if any(_is_kronos(n) for n in todo):         # estimate may not know Kronos
            est = max(est, SYNC_LIMIT + 1)

    if est > SYNC_LIMIT:                             # slow: background worker
        return _out(dict(ok=True, **start_job(bid, job, est)))

    bt = run_backtest_blocking(bid, job)             # fast: inline
    return _out(dict(ok=True, backtest_id=bid, status="done", models=names,
                     h=args.h, order=order, origins=int(bt["origin"].nunique()),
                     first=bt["origin"].min(), last=bt["origin"].max()))

@tool(StatusArgs, slow=True, name="backtest_status")
def backtest_status_tool(args) -> dict:
    """Wait up to wait_s seconds (max 45) for a running backtest. Call repeatedly until status is done."""
    from backtest.jobs import backtest_status
    out = backtest_status(args.backtest_id, args.wait_s)
    if out.get("error") == "UNKNOWN_ID":
        raise ToolError("UNKNOWN_ID", "No such backtest job.", "Call list_artifacts.")
    return clean_json(dict(ok=True, **out))


# ---------- evaluation ----------
@tool(MetricsArgs)
def get_metrics(args) -> dict:
    """Accuracy metrics for one model: MAE, RMSE, dir_acc, coverage, width (log-return units)."""
    bt = _load_bt(args.backtest_id)
    _check_model(bt, args.model)
    s = score(bt[bt["model"] == args.model]).reset_index().iloc[0].to_dict()
    for k in ("ticker", "model", "h"):
        s.pop(k, None)
    meta = _get_meta(args.backtest_id)
    ticker = get_meta(meta.get("ds", "")).get("ticker")
    order = meta.get("order") if args.model == "arima" else None
    return clean_json(dict(ok=True, ticker=ticker, model=args.model, order=order, **s))


@tool(DMArgs)
def dm_test(args) -> dict:
    """Diebold-Mariano test of model_a vs model_b on squared cumulative-return error.

    Positive statistic with p<0.05 means model_a is significantly worse.
    Uses Newey-West and Harvey correction with the backtest's own step.
    """
    if args.model_a == args.model_b:
        raise ToolError("BAD_MODEL", "model_a and model_b are identical.", "Pick two different models.")
    bt = _load_bt(args.backtest_id)
    meta = _get_meta(args.backtest_id)
    for m in (args.model_a, args.model_b):
        _check_model(bt, m)
    h = meta.get("h", args.h)
    results = {m: bt[bt["model"] == m] for m in (args.model_a, args.model_b)}
    tab = dm_table(results, benchmark=args.model_b, h=h, step=meta.get("step", 1))
    a = args.model_a
    row = tab.reset_index().query("model == @a").iloc[0].to_dict()
    row.pop("ticker", None)
    return clean_json(dict(ok=True, model_b=args.model_b, **row))


# (header, score() column, decimals)
TABLE_COLS = [("MAE", "MAE", 5), ("RMSE", "RMSE", 5), ("DirAcc", "dir_acc", 3),
              ("Cov90", "coverage", 3), ("Width", "width", 4)]


def _fmt(v, nd):
    return "-" if v is None or pd.isna(v) else f"{v:.{nd}f}"

@tool(CompareArgs)
def compare_models(args) -> dict:
    """Ready-made markdown table: accuracy metrics and DM p-value vs the benchmark.

    Paste the `table` field verbatim into the answer. Do not reformat the numbers.
    """
    bt = _load_bt(args.backtest_id)
    meta = _get_meta(args.backtest_id)
    bench = args.benchmark
    _check_model(bt, bench)
    h, step = meta.get("h", args.h), meta.get("step", 1)

    models = [bench] + [m for m in bt["model"].unique() if m != bench]
    results = {m: bt[bt["model"] == m] for m in models}
    dm = dm_table(results, benchmark=bench, h=h, step=step).reset_index().set_index("model")

    cw = cw_table(results, benchmark=bench, h=h, step=step)["p"].droplevel("ticker").to_dict() if bench == "naive" else {}
    dd = dm_table({k: v for k, v in results.items() if k != "naive"}, benchmark="drift", h=h, step=step)["p"].droplevel("ticker").to_dict() if "drift" in results and bench != "drift" else None
    header = ["Model"] + [c[0] for c in TABLE_COLS] + ["DM p-value", "CW p-value"] + ([("DM p vs drift")] if dd is not None else [])
    lines = ["| " + " | ".join(header) + " |", "|:--|" + "--:|" * (len(header) - 1)]
    for m in models:
        s = score(results[m]).reset_index().iloc[0].to_dict()
        cells = [_fmt(s.get(key), nd) for _, key, nd in TABLE_COLS]
        p = None if m == bench else float(dm.loc[m, "p"])
        lines.append("| " + " | ".join([m] + cells + [_fmt(p, 3), _fmt(cw.get(m), 3)] + ([_fmt(dd.get(m), 3)] if dd is not None else [])) + " |")

    n = int(bt.loc[bt["model"] == bench, "origin"].nunique())
    ticker = get_meta(meta.get("ds", "")).get("ticker")
    order = meta.get("order")
    note = f"n={n}, h={h}, step={step}. DM p-value is vs {bench}." + (" CW p-value (one-sided, nested models only) is vs naive." if bench == "naive" else "") + (" DM p vs drift uses drift as a reference benchmark." if dd is not None else "")
    if order:
        note += f" ARIMA order={tuple(order)}."
    return clean_json(dict(ok=True, ticker=ticker, table="\n".join(lines), n=n,
                           benchmark=bench, order=order, note=note))

