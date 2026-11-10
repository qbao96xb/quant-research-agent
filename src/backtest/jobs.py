# src/backtest/jobs.py
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "artifacts" / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)

MAX_WAIT_S = 45          # keep equal to StatusArgs.wait_s le=45


def read_state(bid):
    p = JOBS / f"{bid}.json"
    return json.loads(p.read_text()) if p.exists() else None


def write_state(bid, **kw):
    p = JOBS / f"{bid}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({**(read_state(bid) or {}), **kw}))
    tmp.replace(p)


def start_job(bid, args, est):
    s = read_state(bid)
    if s and s["status"] in ("running", "done"):           # idempotent
        return {"status": s["status"], "backtest_id": bid}
    write_state(bid, status="running", started=time.time(),
                done=0, total=len(args["models"]), eta_s=round(est),
                error=None, pid=None)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    with open(JOBS / f"{bid}.log", "a") as log:
        p = subprocess.Popen(
            [sys.executable, "-m", "backtest.worker", bid, json.dumps(args)],
            cwd=ROOT, env=env, start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=log)
    write_state(bid, pid=p.pid)                            # merged into existing state
    return {"status": "running", "backtest_id": bid, "eta_s": round(est),
            "hint": "Call backtest_status(backtest_id) until status is done."}


def cancel_running_jobs():
    """Stop every background backtest marked running. Returns how many."""
    n = 0
    for f in JOBS.glob("*.json"):
        bid = f.stem
        s = read_state(bid)
        if not s or s.get("status") != "running":
            continue
        pid = s.get("pid")
        if pid:
            try:
                os.killpg(pid, signal.SIGTERM)   # worker has its own session
            except (ProcessLookupError, PermissionError):
                pass
        write_state(bid, status="cancelled", error="Cancelled.")
        n += 1
    return n


def backtest_status(backtest_id, wait_s=30):
    end = time.time() + min(wait_s, MAX_WAIT_S)
    while True:
        s = read_state(backtest_id)
        if s is None:
            return {"error": "UNKNOWN_ID"}
        if s["status"] == "running" and time.time() - s["started"] > 900:
            write_state(backtest_id, status="failed",
                        error="Worker timed out or was killed.")
            s = read_state(backtest_id)
        if s["status"] != "running" or time.time() >= end:
            break
        time.sleep(2)
    out = {k: s.get(k) for k in ("status", "done", "total", "eta_s", "error")}
    out["elapsed_s"] = round(time.time() - s["started"])
    if out["status"] == "running":
        out["hint"] = "Still running. Call backtest_status again."
        out["stall_hint"] = ("Still running after several polls. Stop polling and tell "
                             "the user it is still running, with the backtest_id.")
    return out