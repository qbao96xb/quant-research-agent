import json

import pytest
import yaml

from src.agent_eval.aggregate import summarize

CASES = [dict(id="I1", category="invalid", split="dev", prompt="Forecast XXXX",
              must_call=["load_data"], expect_error_code=["BAD_TICKER"])]

LOAD_BAD = ("load_data", {"ok": False, "error": {"code": "BAD_TICKER", "message": "nf"}})
BLOCKED = ("list_artifacts", {"ok": False, "error": {"code": "NO_FALLBACK", "message": "blocked"}})
FC_OK = ("forecast", {"ok": True, "mean": 0.01})


def totals_for(tmp_path, trace):
    run = dict(case_id="I1", category="invalid", run=0, latency=1, answer="Ticker not found.",
               steps=3, finish="done", error=None,
               trace=[dict(tool=t, out=o) for t, o in trace])
    cp = tmp_path / "cases.yaml"
    cp.write_text(yaml.safe_dump(CASES))
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "runs.jsonl").write_text(json.dumps(run) + "\n")
    totals, _, _ = summarize(rd, cases_path=cp)
    return totals


def test_loop_block_is_not_a_tool_failure(tmp_path):
    totals = totals_for(tmp_path, [LOAD_BAD, BLOCKED, FC_OK])
    assert totals["n_calls"] == 1                  # BAD_TICKER expected, NO_FALLBACK blocked
    assert totals["tool_success"] == pytest.approx(1.0)
    assert totals["n_blocked"] == 1


def test_no_calls_gives_zero_blocked(tmp_path):
    totals = totals_for(tmp_path, [])
    assert totals["n_blocked"] == 0