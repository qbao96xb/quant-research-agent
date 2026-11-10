"""Run the phase2 charts notebook, extract its outputs, and draft results/findings.md.

    python scripts/write_findings.py                  # run notebook, then write findings
    python scripts/write_findings.py --skip-notebook  # use outputs already saved in the notebook

Needs: pip install nbformat nbclient ipykernel
"""
import argparse
import base64
import os
import json
import re
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import nbformat
import pandas as pd
import requests
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from rich.console import Console
from rich.progress import (BarColumn, Progress, TaskProgressColumn, TextColumn,
                           TimeElapsedColumn)

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_INTS = {5, 90, 10, 1}   # h, coverage target, numbers of samples, etc.
console = Console()

# Bullet tags -> report headings. The model tags each bullet; the script sorts them.
SECTIONS = OrderedDict([
    ("ACCURACY", "Accuracy vs naive"),
    ("INTERVALS", "Forecast intervals"),
    ("DIRECTION", "Direction and signal"),
    ("VOLATILITY", "GARCH volatility"),
    ("CAVEAT", "Caveats"),
])


# ---- Live progress ---------------------------------------------------------------
class Tracker:
    """Two rich bars: weighted overall progress plus the current step."""

    WEIGHTS = OrderedDict(find=0.01, notebook=0.14, charts=0.02, facts=0.02,
                          generate=0.75, check=0.03, save=0.03)

    def __init__(self, progress, max_tokens):
        self.progress = progress
        self.max_tokens = max(max_tokens, 1)
        self.done = 0.0
        self.shown = 0.0
        self.pulse = None
        self.overall = progress.add_task("Overall", total=1.0)
        self.step = progress.add_task("Starting", total=1.0)

    def _set_overall(self, fraction, label):
        self.shown = max(self.shown, min(fraction, 1.0))
        self.progress.update(self.overall, completed=self.shown,
                             description=f"Overall | {label}")

    def begin(self, label, indeterminate=False):
        if indeterminate:
            self.progress.update(self.step, visible=False)
            self.pulse = self.progress.add_task(label, total=None)
        else:
            self.progress.update(self.step, description=label, completed=0.0,
                                 total=1.0, visible=True)

    def finish(self, key, label):
        if self.pulse is not None:
            self.progress.remove_task(self.pulse)
            self.pulse = None
        self.done += self.WEIGHTS[key]
        self.progress.update(self.step, completed=1.0, visible=True)
        self._set_overall(self.done, label)

    def generating(self, model, phase, n_tok, rate):
        fraction = min(n_tok / self.max_tokens, 0.95)
        self.progress.update(self.step, completed=fraction, description=(
            f"5/7 {model} [{phase}] {n_tok}/{self.max_tokens} tok, {rate:.1f} tok/s"))
        self._set_overall(self.done + self.WEIGHTS["generate"] * fraction, "generating")

    def complete(self):
        self._set_overall(1.0, "done")
        self.progress.update(self.step, completed=1.0, description="Done")


# ---- Timing recorder -------------------------------------------------------------
class Timings:
    def __init__(self):
        self.data = OrderedDict()

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.data[name] = self.data.get(name, 0.0) + time.perf_counter() - start

    def report(self):
        total = sum(self.data.values()) or 1.0
        console.print("\nTimings")
        for name, secs in sorted(self.data.items(), key=lambda x: -x[1]):
            console.print(f"  {name:<22} {secs:8.1f}s  {secs / total:6.1%}")


TIMINGS = Timings()


# ---- Notebook: run it, then read its outputs --------------------------------------
def run_notebook(path, executed_path, timeout):
    """Execute the notebook in its own folder (so '../results' resolves). Original untouched."""
    nb = nbformat.read(path, as_version=4)
    nb.cells.insert(0, nbformat.v4.new_code_cell("%matplotlib inline"))   # force embedded PNGs
    os.environ.pop("MPLBACKEND", None)                                    # don't inherit a GUI backend
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    except CellExecutionError as exc:
        raise RuntimeError(f"Notebook failed:\n{exc}") from exc
    nb.cells.pop(0)                                                       # keep cell numbers unchanged
    nbformat.write(nb, executed_path)
    return nb

def extract_outputs(nb, fig_dir):
    """Collect images and result tables in notebook order. No knowledge of what they show.

    Returns (items, run_name). Each item: dict(kind="image"|"table", cell=int, path|text).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    for old in fig_dir.glob("*.png"):
        old.unlink()
    items, run_name, n_img = [], None, 0
    for cell_no, cell in enumerate((c for c in nb.cells if c.cell_type == "code"), start=1):
        for out in cell.get("outputs", []):
            kind = out.get("output_type")
            if kind == "stream" and run_name is None:
                m = re.search(r"Run:\s*(\S+)", out.get("text", ""))
                if m:
                    run_name = m.group(1)
            data = out.get("data", {})
            if "image/png" in data:
                n_img += 1
                path = fig_dir / f"figure_{n_img:02d}.png"
                path.write_bytes(base64.b64decode(data["image/png"]))
                items.append(dict(kind="image", cell=cell_no, path=path))
            elif kind == "execute_result" and "text/plain" in data:
                items.append(dict(kind="table", cell=cell_no, text=data["text/plain"]))
    return items, run_name


# ---- Data ------------------------------------------------------------------------
def latest_run():
    runs = sorted(p for p in (ROOT / "results" / "phase2").iterdir() if p.name[0].isdigit())
    if not runs:
        raise SystemExit("No phase2 run found. Run run_phase2.py first.")
    return runs[-1]


def build_facts(run):
    acc = pd.read_csv(run / "accuracy.csv", index_col=[0, 1])
    dm = pd.read_csv(run / "dm.csv", index_col=[0, 1])
    vol = pd.read_csv(run / "volatility.csv", index_col=0)
    cfg = json.loads((run / "config.json").read_text())

    t = acc[["n", "MAE", "RMSE", "DirAcc", "Cov90", "Width"]].copy()
    naive = acc["RMSE"].xs("naive", level="model")
    t["RMSE_ratio_vs_naive"] = [r.RMSE / naive[tk] for (tk, _), r in acc.iterrows()]
    t = t.join(dm[["dm", "p", "verdict"]])
    setup = (f"horizon h={cfg['horizon']}, step={cfg['step']}, start={cfg['start']}, "
             f"tickers={cfg['tickers']}, origins per ticker={t['n'].min():.0f}")
    facts = (f"SETUP: {setup}\n\nPER TICKER AND MODEL (h=5; verdict is DM test vs naive, "
             f"p<0.05):\n{t.round(4).to_string()}\n\nGARCH VOLATILITY:\n{vol.round(4).to_string()}")
    return facts, setup


PROMPT = """You are summarizing a forecasting backtest. Use ONLY the facts below.

Write 5 to 8 markdown bullets (each starting with "- ").
Start every bullet with exactly one tag in square brackets, then the text:
- [ACCURACY] for items (1) and (2)
- [INTERVALS] for item (3)
- [DIRECTION] for item (4)
- [VOLATILITY] for item (5)
- [CAVEAT] for item (6)
Write at least one bullet per tag, in the order above.

Rules:
- Copy numbers exactly as shown. Never compute, round, or invent numbers.
- Do not use percentages, only the decimals shown.
- Name the ticker and model for each number.
- Items to cover: (1) which models beat or lose to naive per the DM verdict and p values,
  (2) RMSE_ratio_vs_naive for the worst and best Kronos variants,
  (3) Cov90 of Kronos vs the 0.90 target and the Width comparison,
  (4) DirAcc versus 0.50, noting it is near chance,
  (5) GARCH predicted vs realized sigma,
  (6) one caveat about sample size (n origins, 2 tickers).
- No introduction, no conclusion, no advice. Bullets only.

FACTS:
{facts}
"""


def numbers(text):
    return re.findall(r"-?\d+\.\d+|-?\d+", text)


def check(bullets, facts):
    pool = [float(x) for x in numbers(facts)]
    bad = []
    for s in numbers(bullets):
        x, dec = float(s), (len(s.split(".")[1]) if "." in s else 0)
        tol = 0.5 * 10 ** -dec + 1e-9
        if dec == 0 and int(x) in ALLOWED_INTS:
            continue
        if not any(abs(x - f) <= tol for f in pool):
            bad.append(s)
    return bad


def ask(model, prompt, host, tracker, max_tokens, think):
    """Stream the reply and feed live token stats to the tracker."""
    content, n_tok, t0, truncated = [], 0, time.time(), False
    with requests.post(f"{host}/api/chat", timeout=600, stream=True, json=dict(
            model=model, stream=True, think=think,
            messages=[{"role": "user", "content": prompt}],
            options=dict(temperature=0, seed=42, num_ctx=8192,
                         num_predict=max_tokens))) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            msg = chunk.get("message", {})
            piece = msg.get("content", "")
            if piece:
                content.append(piece)
            if piece or msg.get("thinking"):
                n_tok += 1
                phase = "writing" if content else "thinking"
                rate = n_tok / max(time.time() - t0, 1e-6)
                tracker.generating(model, phase, n_tok, rate)
            if chunk.get("done"):
                n_tok = chunk.get("eval_count", n_tok)
                truncated = chunk.get("done_reason") == "length"
                break
    return "".join(content).strip(), n_tok, truncated


# ---- Report ----------------------------------------------------------------------
BULLET_RE = re.compile(r"^-\s*\[([A-Za-z]+)\]\s*(.+)$")


def group_bullets(bullets):
    groups = {tag: [] for tag in SECTIONS}
    other = []
    for line in bullets.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        m = BULLET_RE.match(line)
        if m and m.group(1).upper() in groups:
            groups[m.group(1).upper()].append(m.group(2).strip())
        else:
            other.append(line[2:].strip())
    return groups, other


def build_report(run, model, status, reasons, setup, groups, other, items, facts, base_dir):
    lines = [
        f"<!-- draft from {model}, run {run.name}, check: {status} -->",
        "<!-- REVIEW BY HAND, then delete these lines and treat as ground truth -->",
        "",
        f"# Phase 2 findings: run {run.name}",
        "",
        f"> **Setup:** {setup}",
        f"> **Check:** {status}" + (f" ({'; '.join(reasons)})" if reasons else ""),
        f"> **Draft by:** {model}. Verify every claim against the figures and the facts table.",
        "",
        "## Summary",
        "",
    ]
    for tag, title in SECTIONS.items():
        lines += [f"### {title}", ""]
        lines += [f"- {b}" for b in groups[tag]] or ["- (no bullet generated)"]
        lines.append("")
    if other:
        lines += ["### Other", ""] + [f"- {b}" for b in other] + [""]

    lines += ["## Figures", "",
              "*Taken from the notebook outputs, in notebook order.*", ""]
    n_img = 0
    for it in items:
        if it["kind"] == "image":
            n_img += 1
            rel = it["path"].relative_to(base_dir).as_posix()
            lines += [f"![Figure {n_img}]({rel})", "",
                      f"*Figure {n_img} (notebook cell {it['cell']})*", ""]
        else:
            lines += [f"**Table from notebook cell {it['cell']}**", "",
                      "```text", it["text"], "```", ""]
    if not items:
        lines += ["*No outputs found in the notebook.*", ""]

    lines += ["## Appendix: facts given to the model", "", "<details>",
              "<summary>Show facts table</summary>", "", "```text", facts, "```", "",
              "</details>", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--notebook", default="notebooks/phase2_charts.ipynb",
                    help="Notebook to execute first (path relative to the project root).")
    ap.add_argument("--notebook-timeout", type=int, default=900,
                    help="Per-cell timeout in seconds.")
    ap.add_argument("--skip-notebook", action="store_true",
                    help="Don't execute the notebook. Read the outputs already saved in it.")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Output cap, also the 100%% mark of the generation bar. "
                         "Thinking tokens count, so keep it generous.")
    ap.add_argument("--think", action="store_true",
                    help="Let the model reason before answering (slow, needs a large --max-tokens).")
    a = ap.parse_args(argv)
    if a.max_tokens is None:
        a.max_tokens = 8000 if a.think else 1500

    nb_path = (ROOT / a.notebook).resolve()
    results = ROOT / "results"
    executed = results / "phase2_charts_executed.ipynb"
    if not nb_path.exists():
        raise SystemExit(f"Notebook not found: {nb_path}. Set --notebook.")

    progress = Progress(
        TextColumn("{task.description:<62}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=10,
    )
    warnings = []
    with progress:
        tracker = Tracker(progress, a.max_tokens)

        tracker.begin("1/7 Finding latest phase2 run")
        with TIMINGS.stage("1_find_run"):
            run = latest_run()
        tracker.finish("find", "found run")

        if a.skip_notebook:
            tracker.begin("2/7 Reading saved notebook outputs")
            with TIMINGS.stage("2_notebook"):
                nb = nbformat.read(nb_path, as_version=4)
        else:
            tracker.begin(f"2/7 Executing {nb_path.name}", indeterminate=True)
            with TIMINGS.stage("2_notebook"):
                nb = run_notebook(nb_path, executed, a.notebook_timeout)
        tracker.finish("notebook", "notebook done")

        tracker.begin("3/7 Extracting notebook outputs")
        with TIMINGS.stage("3_extract"):
            items, nb_run = extract_outputs(nb, results / "figures" / run.name)
            if not any(i["kind"] == "image" for i in items):
                warnings.append("No images found in the notebook outputs.")
            if nb_run and nb_run != run.name:
                warnings.append(f"Notebook used run {nb_run} but facts use {run.name}.")
        tracker.finish("charts", f"{sum(i['kind'] == 'image' for i in items)} figures")

        tracker.begin(f"4/7 Building facts table from {run.name}")
        with TIMINGS.stage("4_build_facts"):
            facts, setup = build_facts(run)
        tracker.finish("facts", "facts ready")

        tracker.begin(f"5/7 {a.model} [loading] waiting for first token")
        with TIMINGS.stage("5_generate"):
            bullets, n_tok, truncated = ask(a.model, PROMPT.format(facts=facts),
                                            a.host, tracker, a.max_tokens, a.think)
        tracker.finish("generate", "generated")

        if not bullets:
            progress.stop()
            console.print(f"[red]Model returned no answer text ({n_tok} tokens used). "
                          f"Nothing saved. Try without --think, or raise --max-tokens.[/red]")
            raise SystemExit(1)

        tracker.begin("6/7 Checking numbers and structure")
        with TIMINGS.stage("6_check"):
            bad = check(bullets, facts)
            groups, other = group_bullets(bullets)
            n_bullets = sum(l.startswith("- ") for l in bullets.splitlines())
            reasons = []
            if truncated:
                reasons.append("output cut off by --max-tokens")
            if bad:
                reasons.append(f"{len(bad)} numbers not found in facts")
            if not 5 <= n_bullets <= 8:
                reasons.append(f"{n_bullets} bullets, expected 5 to 8")
            missing = [t for t, v in groups.items() if not v]
            if missing:
                reasons.append("no bullet for: " + ", ".join(missing))
            if other:
                reasons.append(f"{len(other)} untagged bullets")
            status = "UNVERIFIED" if reasons else "numbers match facts"
        tracker.finish("check", "checked")

        tracker.begin("7/7 Saving report")
        with TIMINGS.stage("7_save"):
            out = results / "findings.md"
            out.write_text(build_report(run, a.model, status, reasons, setup, groups,
                                        other, items, facts, results))
            (results / "findings_facts.txt").write_text(facts)
        tracker.finish("save", "saved")
        if status == "numbers match facts":
            tracker.complete()

    console.print("\n" + bullets)
    color = "green" if status == "numbers match facts" else "red"
    console.print(f"\nbullets={n_bullets}, tokens={n_tok}, status=[{color}]{status}[/{color}]")
    for r in reasons:
        console.print(f"[red]- {r}[/red]")
    if bad:
        console.print("Numbers not found in facts:", bad)
    for w in warnings:
        console.print(f"[yellow]Warning: {w}[/yellow]")
    TIMINGS.report()
    console.print(f"Saved {out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\nStopped by user.")
        raise SystemExit(130)