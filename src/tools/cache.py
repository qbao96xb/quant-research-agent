# src/tools/cache.py
import hashlib, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from .errors import ToolError
from .registry_core import tool
from .schemas import ListArgs

STORE = Path(__file__).resolve().parents[2] / "data" / "cache" / "artifacts"
STORE.mkdir(parents=True, exist_ok=True)
_MEM = {}

def data_fp(df):
    h = pd.util.hash_pandas_object(df[["timestamps", "close"]], index=False)
    return hashlib.sha1(h.to_numpy().tobytes()).hexdigest()[:12]

def make_id(prefix, payload):
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return f"{prefix}_{h}"

def put_as(aid, kind, obj, payload=None):
    """Like put, but saves under a fixed ID (needed for background jobs)."""
    if not str(aid).startswith(kind + "_"):
        raise ValueError(f"ID {aid!r} must start with '{kind}_'")
    tmp = STORE / f"{aid}.tmp"
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(tmp)
        tmp.replace(STORE / f"{aid}.parquet")
    else:
        tmp.write_text(json.dumps(obj))
        tmp.replace(STORE / f"{aid}.json")
    if kind == "ds" and isinstance(payload, dict):
        (STORE / f"{aid}.meta.json").write_text(json.dumps(payload, default=str))
    _MEM[aid] = obj
    return aid

def get_meta(aid):
    p = STORE / f"{aid}.meta.json"
    return json.loads(p.read_text()) if p.exists() else {}

def put(kind, obj, payload):
    return put_as(make_id(kind, payload), kind, obj, payload)


def get(aid, kind=None):
    if kind and not str(aid).startswith(kind + "_"):
        raise ToolError("UNKNOWN_ID", f"{aid!r} is not a {kind} ID",
                        f"Expected an ID starting with '{kind}_'.")
    if aid in _MEM:
        return _MEM[aid]
    for ext in ("parquet", "json"):
        p = STORE / f"{aid}.{ext}"
        if p.exists():
            obj = pd.read_parquet(p) if ext == "parquet" else json.loads(p.read_text())
            _MEM[aid] = obj
            return obj
    raise ToolError("UNKNOWN_ID", f"No artifact {aid!r}",
                    "Call load_data or run_backtest first, or use list_artifacts.")


def get_dataset(aid):
    return get(aid, "ds")


def clean_json(x):
    """Make values JSON safe: numpy -> python, NaN -> None, floats rounded."""
    if isinstance(x, dict):
        return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(f"{float(x):.6g}")
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(x).date())
    return x


@tool(ListArgs, name="list_artifacts")
def list_artifacts(args=None):
    """List saved datasets, forecasts and backtest results, newest first.

    Use when the user refers to earlier data or results without naming an ID.
    If several datasets exist, the response includes a note on which one to
    assume and confirm with the user.
    """
    entries = sorted(
        (p for p in STORE.iterdir()
         if p.suffix in (".parquet", ".json") and not p.stem.endswith(".meta")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    artifacts = [
        {"id": p.stem,
         "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))}
        for p in entries
    ]

    datasets = [a for a in artifacts if a["id"].startswith("ds_")]
    out = {"ok": True, "artifacts": artifacts, "datasets": len(datasets)}
    if len(datasets) > 1:
        out["note"] = (
            f"{len(datasets)} datasets are saved. The user did not name one. "
            f"Pick the most recent dataset ({datasets[0]['id']}), begin your answer with "
            f"'I assumed {datasets[0]['id']} (the most recent of {len(datasets)} datasets).', "
            "and end by asking the user to confirm or name another."
        )
    return out
