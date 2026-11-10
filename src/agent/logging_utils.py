import json, time
from datetime import date
from pathlib import Path

LOGS = Path(__file__).resolve().parents[2] / "logs"

def log_event(**kw):
    LOGS.mkdir(exist_ok=True)
    kw["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(LOGS / f"agent_{date.today():%Y%m%d}.jsonl", "a") as f:
        f.write(json.dumps(kw, default=str) + "\n")