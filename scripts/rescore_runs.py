"""Re-score saved eval runs with the current scorer and cases. Prints the runs whose result changed; writes nothing.

Usage: python scripts/rescore_runs.py RUN_DIR [RUN_DIR ...] [--cases src/agent_eval/cases_v2.yaml]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from src.agent_eval.aggregate import load_cases  # noqa: E402
from src.agent_eval.paths import CASES_PATH  # noqa: E402
from src.agent_eval.scoring import score  # noqa: E402


def rescore(run_dir, cases):
    """(case, run, old_passed, new_passed, new_fails) for every saved run in run_dir."""
    old = {}
    with open(Path(run_dir) / "scored.csv") as f:
        for row in csv.DictReader(f):
            old[(row["case"], int(row["run"]))] = row["passed"] == "True"
    rows = []
    for line in open(Path(run_dir) / "runs.jsonl"):
        r = json.loads(line)
        s = score(cases[r["case_id"]], r)
        rows.append((r["case_id"], r["run"], old.get((r["case_id"], r["run"])), s["passed"], ",".join(s["fails"]))
        )
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--cases", default=str(CASES_PATH))
    a = ap.parse_args(argv)
    cases = load_cases(a.cases)
    for d in a.run_dirs:
        rows = rescore(d, cases)
        changed = [x for x in rows if x[2] != x[3]]
        old_n = sum(bool(x[2]) for x in rows)
        new_n = sum(bool(x[3]) for x in rows)
        print(f"== {d}: old {old_n}/{len(rows)}  new {new_n}/{len(rows)}  changed {len(changed)}")
        for c, n, o, p, fails in changed:
            print(f"  {c} run {n}: {'pass' if o else 'fail'} -> {'pass' if p else 'fail'} {fails}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
