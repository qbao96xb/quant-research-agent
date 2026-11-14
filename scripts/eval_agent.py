"""Run the frozen eval for ONE model and write the scored results.

PYTHONPATH=src:. python -m scripts.eval_agent --name qwen3.5-9b --model qwen3.5:9b --n 3
PYTHONPATH=src:. python -m scripts.eval_agent --name qwen3.5-9b --model qwen3.5:9b --only S01 A01 --n 1
PYTHONPATH=src:. python -m scripts.eval_agent --name qwen3.5-9b --model qwen3.5:9b --split test   # final report only
"""
import argparse
import json
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.progress import (BarColumn, MofNCompleteColumn, Progress as RichProgress,
                           TaskProgressColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from rich.table import Table

from src.agent.llm import OpenAICompatLLM
from src.agent_eval.aggregate import load_cases, summarize
from src.agent_eval.scoring import score
from src.agent_eval.runner import run_all
from src.agent_eval.paths import CASES_PATH
from src.data.asof import DATA_ASOF, set_asof
from src.backtest.jobs import cancel_running_jobs

EXECUTOR_LOCK = Path(__file__).resolve().parents[1] / "handoff" / ".executor.lock"


def _git(*args):
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
    except Exception:
        return ""
    return out.stdout.strip()


def git_info():
    """Commit, branch and dirty flag, so an eval run names the code it tested."""
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return dict(commit=_git("rev-parse", "--short", "HEAD"), branch=_git("branch", "--show-current"), dirty=bool(dirty))


def case_ids(split, cases_path=CASES_PATH):
    cases = load_cases(cases_path)
    return [i for i, c in cases.items() if split == "all" or c.get("split", "dev") == split]


class Progress:
    """Rich live display: total bar, per-case bar and a table of recent runs."""

    def __init__(self, cases_path=CASES_PATH, keep=6):
        self.cases = load_cases(cases_path)
        self.recent = deque(maxlen=keep)
        self.passed = self.finished = 0
        self.total_id = self.case_id = None
        self.bars = RichProgress(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[status]}"),
            expand=True)
        self.live = Live(self._render(), refresh_per_second=4)

    def _render(self):
        t = Table(expand=True, show_edge=False, title="Recent runs")
        for col in ("Case", "Run", "Result", "Latency", "Steps"):
            t.add_column(col)
        for case, run, ok, label, lat, steps in self.recent:
            style = "green" if ok else "red"
            t.add_row(case, str(run), f"[{style}]{label}[/]", f"{lat:.0f}s", str(steps))
        return Group(self.bars, t)

    def start(self):
        self.live.start()

    def close(self):
        self.live.stop()

    def __call__(self, kind, **info):
        if kind == "start":
            self.total_id = self.bars.add_task(
                "Total", total=info["total"], completed=info["done"], status="")
        elif kind == "case_start":
            if self.case_id is not None:
                self.bars.remove_task(self.case_id)
            c = info["case"]
            desc = f"{c['id']:<5}{c['prompt'].strip()[:40]!r}"
            self.case_id = self.bars.add_task(
                desc, total=info["n"], completed=info["done"], status="")
        elif kind == "run_done":
            c, r = info["case"], info["run"]
            s = score(self.cases[c["id"]], r)
            self.finished += 1
            self.passed += s["passed"]
            label = "PASS" if s["passed"] else "FAIL " + ",".join(s["fails"])
            self.recent.append((c["id"], r["run"], s["passed"], label, r["latency"], r["steps"]))
            self.bars.update(self.case_id, advance=1, status=label)
            self.bars.update(self.total_id, advance=1,
                             status=f"pass {self.passed}/{self.finished} "
                                    f"({self.passed / self.finished:.0%})")
        self.live.update(self._render())

def _isolated(bar):
    """Forward events to the progress bar and kill leftover jobs after every run."""
    def cb(kind, **info):
        if bar:
            bar(kind, **info)
        if kind == "run_done":
            cancel_running_jobs()
    return cb

def evaluate(llm, out, n=3, split="dev", only=None, resume=False, meta=None, progress=True, cases_path=CASES_PATH):
    if EXECUTOR_LOCK.exists():
        raise SystemExit(f"executor run active ({EXECUTOR_LOCK}); wait for it to finish or remove a stale lock")
    ids = set(only) if only else set(case_ids(split, cases_path))
    bar = Progress(cases_path) if progress else None
    cancel_running_jobs()                      # clean start
    try:
        if bar:
            bar.start()
        run_all(llm, out, n=n, only=ids, cases_path=cases_path, resume=resume, on_event=_isolated(bar))
    finally:
        cancel_running_jobs()                  # clean end, even after Ctrl-C
        if bar:
            bar.close()
    totals, df, tables = summarize(out, cases_path)

    out = Path(out)
    (out / "summary.json").write_text(
        json.dumps(dict(meta=meta or {}, totals=totals), indent=2, default=float))
    df.to_csv(out / "scored.csv", index=False)
    for k, v in tables.items():
        v.to_csv(out / f"{k}.csv")
    return totals, df, tables


def print_report(totals, tables):
    print("\n== Totals ==")
    for k, v in totals.items():
        print(f"{k:>18}: {v:.3f}" if isinstance(v, float) else f"{k:>18}: {v}")
    for k in ("by_category", "by_split", "latency_by_cat", "labels"):
        print(f"\n== {k} ==\n{tables[k].to_string()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="label used in the folder name")
    ap.add_argument("--model", required=True, help="Ollama model name passed to OpenAICompatLLM")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--only", nargs="*", help="case ids, overrides --split")
    ap.add_argument("--prompt-ver", default="v2")
    ap.add_argument("--out", help="fixed folder (needed for --resume)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--cases", default=str(CASES_PATH), help="cases YAML (default: frozen cases.yaml)")
    ap.add_argument("--data-asof", default=DATA_ASOF, help="cap price data at this date; 'none' = latest data")
    a = ap.parse_args()
    asof = set_asof(a.data_asof)

    out = a.out or f"results/eval/{a.name}_{a.prompt_ver}_{a.split}_{datetime.now():%Y%m%d_%H%M}"
    meta = dict(**git_info(), name=a.name, model=a.model, n=a.n, split=a.split,
                only=a.only, prompt_ver=a.prompt_ver, cases=a.cases, data_asof=asof, started=f"{datetime.now():%Y-%m-%d %H:%M}")
    totals, _, tables = evaluate(OpenAICompatLLM(model=a.model), out, n=a.n, split=a.split,
                                 only=a.only, resume=a.resume, meta=meta, cases_path=a.cases,
                                 progress=not a.no_progress)
    print_report(totals, tables)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
