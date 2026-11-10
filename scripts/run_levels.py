import json, subprocess, sys, time
from collections import Counter
from datetime import datetime
from pathlib import Path
import re

from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           TextColumn, TimeElapsedColumn, TimeRemainingColumn,
                           SpinnerColumn)
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
RUNS = 5
console = Console()

# (level, query, required tool counts, forbidden tools, text that must appear)
CASES = [
    (1, "Load KO data since 2015 and describe it",
        {"load_data": 1, "describe_data": 1}, [], ""),
    (1, "What is the 5-day GARCH volatility for KO?",
        {"load_data": 1, "garch_volatility": 1}, [], ""),
    (1, "List my saved artifacts",
        {"list_artifacts": 1}, ["load_data"], ""),
    (2, "Forecast KO for 5 days with ARIMA",
        {"load_data": 1, "forecast": 1}, ["run_backtest"], ""),
    (2, "Backtest ARIMA on KO and give metrics",
        {"load_data": 1, "run_backtest": 1, "get_metrics": 1}, [], ""),
    (3, "Forecast KO for 5 days and compare to naive",
     {"load_data": 1, "forecast": 1, "run_backtest": 1, "compare_models": 1}, [], ""),
    (4, "Which model is best for KO?",
     {"load_data": 1, "run_backtest": 1, "compare_models": 1}, [], ""),
    (5, "Forecast XXXX for 5 days",
        {"load_data": 1}, ["forecast", "run_backtest"], "BAD_TICKER|DATA_FETCH_FAILED|could not|check the symbol"),
    # Level 6: boundaries and grounding
    (6, "Forecast it",
     {}, ["load_data", "forecast", "run_backtest"], ""),
    (6, "Compute the Sharpe ratio for KO",
     {}, ["run_backtest", "get_metrics", "dm_test"], "not available|not supported|cannot|can't|isn't"),
    (6, "Describe the dataset I loaded earlier",
     {"list_artifacts": 1}, ["load_data"], ""),
    (6, "Is ARIMA significantly better than naive for KO?",
     {"load_data": 1, "run_backtest": 1, "dm_test": 1}, [], "p-value"),
    (6, "Delete all my artifacts",
     {}, ["load_data", "forecast", "run_backtest"], ""),
]

def log_path():
    return ROOT / "logs" / f"agent_{datetime.now():%Y%m%d}.jsonl"

def read_lines(p):
    return p.read_text().splitlines() if p.exists() else []

def tools_in(lines):
    names = []
    for ln in lines:
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        name = rec.get("tool") or rec.get("name")  # adjust to your log schema
        if name:
            names.append(name)
    return Counter(names)

def run_once(query, required, forbidden, must_contain):
    p = log_path()
    before = len(read_lines(p))
    try:
        out = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_agent.py"), query],
            capture_output=True, text=True, cwd=ROOT, timeout=240,
        )
    except subprocess.TimeoutExpired:
        return False, {"_error": "subprocess timeout"}
    new = tools_in(read_lines(p)[before:])
    ok = all(new[t] >= n for t, n in required.items())
    ok &= not any(new[t] for t in forbidden)
    blob = out.stdout + out.stderr + json.dumps(read_lines(p)[before:])
    ok &= bool(re.search(must_contain, blob, re.I)) if must_contain else True
    return ok, dict(new)

if __name__ == "__main__":
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    cases = [c for c in CASES if not only or c[0] == only]
    summary = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("starting", total=len(cases) * RUNS)
        for level, q, req, forb, text in cases:
            results = []
            for i in range(RUNS):
                progress.update(
                    task, description=f"L{level} run {i+1}/{RUNS} | {q[:45]}")
                t0 = time.time()
                ok, calls = run_once(q, req, forb, text)
                results.append((ok, calls))
                mark = "[green]PASS[/]" if ok else "[red]FAIL[/]"
                progress.console.print(
                    f"L{level} run {i+1}/{RUNS} {mark} "
                    f"({time.time() - t0:.0f}s) {q[:40]}")
                if not ok:
                    progress.console.print(f"    [dim]calls={calls}[/]")
                progress.advance(task)
            summary.append((level, q, sum(r[0] for r in results)))

    table = Table(title="Results")
    table.add_column("Level", justify="right")
    table.add_column("Query")
    table.add_column("Passed", justify="right")
    for level, q, passed in summary:
        color = "green" if passed == RUNS else ("yellow" if passed else "red")
        table.add_row(str(level), q, f"[{color}]{passed}/{RUNS}[/]")
    console.print(table)
