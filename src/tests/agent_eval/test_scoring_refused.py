"""max_calls counts only calls that did work; calls refused before running do not count (owner, 2026-11-14)."""
from src.agent_eval.scoring import score


def case(**kw):
    base = dict(id="T", category="tuning", prompt="Tune ARIMA on AAPL", max_steps=8)
    base.update(kw)
    return base


def run(trace, answer="ok"):
    return dict(answer=answer, finish="done", steps=2, latency=1.0,
                trace=[dict(tool=t, out=o) for t, o in trace])


def err(code):
    return {"ok": False, "error": {"code": code, "message": "refused"}}


OK_TUNE = ("tune_arima", {"ok": True})


def test_refused_calls_do_not_count_toward_limit():
    t = [("tune_arima", err("BAD_ARGUMENT")), ("tune_arima", err("OVER_BUDGET")), OK_TUNE]
    assert score(case(max_calls={"tune_arima": 1}), run(t))["passed"]


def test_two_real_calls_still_fail():
    s = score(case(max_calls={"tune_arima": 1}), run([OK_TUNE, OK_TUNE]))
    assert s["fails"] == ["WRONG_TOOL"]


def test_call_cap_block_counts():
    t = [OK_TUNE, ("tune_arima", err("CALL_CAP"))]
    assert "WRONG_TOOL" in score(case(max_calls={"tune_arima": 1}), run(t))["fails"]


def test_unparseable_output_counts():
    t = [OK_TUNE, ("tune_arima", err("UNPARSEABLE"))]
    assert "WRONG_TOOL" in score(case(max_calls={"tune_arima": 1}), run(t))["fails"]


def test_zero_limit_counts_refused_attempts():
    t = [("forecast", err("BAD_ARGUMENT"))]
    assert "WRONG_TOOL" in score(case(max_calls={"forecast": 0}), run(t))["fails"]
