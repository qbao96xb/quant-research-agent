import json
from pathlib import Path

import pandas as pd
import yaml

from src.agent_eval.scoring import score, ARG_ERRORS
from src.agent_eval.paths import CASES_PATH


def load_cases(path=CASES_PATH):
    return {c["id"]: c for c in yaml.safe_load(Path(path).read_text())}


def _safe_mean(s):
    return float(s.mean()) if len(s) else float("nan")


def summarize(run_dir, cases_path=CASES_PATH):
    cases = load_cases(cases_path)
    rows, calls = [], []
    for line in open(Path(run_dir) / "runs.jsonl"):
        r = json.loads(line)
        case = cases[r["case_id"]]
        s = score(case, r)
        rows.append(dict(
            case=r["case_id"], cat=r["category"], split=case.get("split", "dev"),
            run=r["run"], passed=s["passed"],
            fails=",".join(s["fails"]), diag=",".join(s["diag"]),
            n_bad=len(s["bad_numbers"]), has_answer=bool(r["answer"].strip()),
            latency=r["latency"], steps=r["steps"], finish=r["finish"]))
        expected = set(case.get("expect_error_code", []))
        for t in r["trace"]:
            out = t["out"] if isinstance(t["out"], dict) else {}
            code = (out.get("error") or {}).get("code")
            calls.append(dict(ok=bool(out.get("ok")),
                              expected_err=code in expected,
                              invalid=code in ARG_ERRORS))

    df, c = pd.DataFrame(rows), pd.DataFrame(calls)
    if len(c):
        c = c[~c.expected_err]                      # expected errors are not failures
    per_case = df.groupby("case").passed.agg(["mean", "any", "all"])
    ans = df[df.has_answer]

    totals = dict(
        tool_success=_safe_mean(c.ok) if len(c) else float("nan"),
        valid_call=1 - _safe_mean(c.invalid) if len(c) else float("nan"),
        completion=df.passed.mean(),
        pass_at_n=per_case["any"].mean(), pass_all_n=per_case["all"].mean(),
        halluc_rate=_safe_mean(ans.n_bad > 0),
        halluc_per_answer=_safe_mean(ans.n_bad),
        lat_median=df.latency.median(), lat_p90=df.latency.quantile(0.9),
        steps_median=df.steps.median(), n_runs=len(df), n_calls=len(c))

    labels = (df.fails + "," + df.diag).str.split(",").explode()
    labels = labels.replace("", pd.NA).dropna()
    tables = dict(
        by_category=df.groupby("cat").passed.mean().round(3),
        by_split=df.groupby("split").passed.mean().round(3),
        latency_by_cat=df.groupby("cat").latency.agg(["median", lambda x: x.quantile(0.9)])
                         .set_axis(["median", "p90"], axis=1).round(1),
        labels=labels.value_counts(),
    )
    return totals, df, tables
