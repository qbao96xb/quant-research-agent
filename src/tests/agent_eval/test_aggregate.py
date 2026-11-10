import json
import math

import pytest
import yaml

from src.agent_eval.aggregate import summarize

CASES = [
    dict(id="S1", category="simple", split="dev", prompt="Forecast KO",
         must_call=["forecast"]),
    dict(id="I1", category="invalid", split="test", prompt="Forecast XXXX",
         must_call=["load_data"], expect_error_code=["BAD_TICKER"],
         answer_must_match=["(?i)not found"]),
    dict(id="A1", category="ambiguous", split="dev", prompt="Forecast it",
         expect="clarify"),
]


def rec(case_id, cat, run, answer, trace, latency, steps=2, finish="done"):
    return dict(case_id=case_id, category=cat, run=run, latency=latency, answer=answer,
                steps=steps, finish=finish, error=None,
                trace=[dict(tool=t, out=o) for t, o in trace])


def write(tmp_path, runs, cases=CASES):
    cp = tmp_path / "cases.yaml"
    cp.write_text(yaml.safe_dump(cases))
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in runs) + "\n")
    return rd, cp


FC_OK = ("forecast", {"ok": True, "mean": 0.01})
FC_BAD = ("forecast", {"ok": False, "error": {"code": "BAD_ARGUMENT", "message": "bad"}})
LOAD_BAD = ("load_data", {"ok": False, "error": {"code": "BAD_TICKER", "message": "nf"}})

RUNS = [
    rec("S1", "simple", 0, "KO forecast mean 0.01", [FC_OK], 10, steps=2),
    rec("S1", "simple", 1, "I think 0.5", [FC_BAD], 20, steps=4),      # hallucinated + bad args
    rec("I1", "invalid", 0, "Ticker not found.", [LOAD_BAD], 5, steps=2),
]


@pytest.fixture
def result(tmp_path):
    rd, cp = write(tmp_path, RUNS)
    return summarize(rd, cases_path=cp)


def test_return_shape(result):
    totals, df, tables = result
    assert len(df) == 3 and set(tables) == {"by_category", "by_split", "latency_by_cat", "labels"}


def test_completion_and_counts(result):
    totals, _, _ = result
    assert totals["n_runs"] == 3
    assert totals["completion"] == pytest.approx(2 / 3)


def test_expected_errors_excluded_from_tool_success(result):
    totals, _, _ = result
    assert totals["n_calls"] == 2                  # BAD_TICKER call dropped
    assert totals["tool_success"] == pytest.approx(0.5)
    assert totals["valid_call"] == pytest.approx(0.5)


def test_pass_at_n_and_pass_all_n(result):
    totals, _, _ = result
    assert totals["pass_at_n"] == pytest.approx(1.0)       # S1 any, I1 any
    assert totals["pass_all_n"] == pytest.approx(0.5)      # only I1 all


def test_hallucination_metrics(result):
    totals, df, _ = result
    assert totals["halluc_rate"] == pytest.approx(1 / 3)
    assert totals["halluc_per_answer"] == pytest.approx(1 / 3)
    assert df.loc[df.fails.str.contains("HALLUCINATED"), "n_bad"].iloc[0] == 1


def test_latency_and_steps(result):
    totals, _, _ = result
    assert totals["lat_median"] == pytest.approx(10)
    assert totals["lat_p90"] == pytest.approx(18)          # linear interpolation of [5, 10, 20]
    assert totals["steps_median"] == pytest.approx(2)


def test_breakdown_tables(result):
    _, _, t = result
    assert t["by_category"]["simple"] == pytest.approx(0.5)
    assert t["by_category"]["invalid"] == pytest.approx(1.0)
    assert t["by_split"]["dev"] == pytest.approx(0.5)
    assert t["by_split"]["test"] == pytest.approx(1.0)
    assert list(t["latency_by_cat"].columns) == ["median", "p90"]


def test_label_counts_include_diag(result):
    _, _, t = result
    assert t["labels"]["HALLUCINATED"] == 1
    assert t["labels"]["BAD_ARGS"] == 1


def test_df_columns(result):
    _, df, _ = result
    assert {"case", "cat", "split", "run", "passed", "fails", "diag", "n_bad",
            "has_answer", "latency", "steps", "finish"} <= set(df.columns)


def test_split_defaults_to_dev(tmp_path):
    cases = [dict(id="S1", category="simple", prompt="Forecast KO")]    # no split field
    rd, cp = write(tmp_path, [rec("S1", "simple", 0, "ok", [], 1)], cases)
    _, df, _ = summarize(rd, cases_path=cp)
    assert df.split.iloc[0] == "dev"


def test_no_tool_calls_does_not_crash(tmp_path):
    runs = [rec("A1", "ambiguous", 0, "Which ticker?", [], 2, steps=1)]
    rd, cp = write(tmp_path, runs)
    totals, df, tables = summarize(rd, cases_path=cp)
    assert math.isnan(totals["tool_success"]) and math.isnan(totals["valid_call"])
    assert totals["completion"] == 1.0 and totals["n_calls"] == 0
    assert len(tables["labels"]) == 0


def test_no_answers_does_not_crash(tmp_path):
    runs = [rec("A1", "ambiguous", 0, "", [], 2, steps=1, finish="max_steps")]
    rd, cp = write(tmp_path, runs)
    totals, _, tables = summarize(rd, cases_path=cp)
    assert math.isnan(totals["halluc_rate"])
    assert totals["completion"] == 0.0
    assert tables["labels"]["LOOP"] == 1 and tables["labels"]["NO_ANSWER"] == 1


def test_unknown_case_id_raises(tmp_path):
    rd, cp = write(tmp_path, [rec("ZZ", "simple", 0, "x", [], 1)])
    with pytest.raises(KeyError):
        summarize(rd, cases_path=cp)
