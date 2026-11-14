from src.agent.scope import CAPPED_TOOLS, cap_block, count_call
from src.agent_eval.aggregate import LOOP_BLOCKS

OK = {"ok": True}

def test_capped_tools():
    assert CAPPED_TOOLS == {"tune_arima", "make_report", "run_backtest",
                            "forecast_vs_actual", "tune_kronos"}

def test_first_call_allowed_second_blocked():
    done = {}
    assert cap_block("make_report", done) is None
    count_call("make_report", OK, OK, done)
    result, meta = cap_block("make_report", done)
    assert result["error"]["code"] == "CALL_CAP"
    assert meta["blocked"] is True

def test_uncapped_tool_never_blocked():
    assert cap_block("forecast", {"forecast": 5}) is None

def test_errors_do_not_count():
    done = {}
    bad = {"ok": False, "error": {"code": "OVER_BUDGET", "message": "x"}}
    count_call("tune_arima", bad, {"ok": False, "error": "OVER_BUDGET"}, done)
    assert cap_block("tune_arima", done) is None

def test_timeout_counts():
    done = {}
    count_call("tune_arima", {"error": {"code": "TIMEOUT", "message": "x"}}, {"ok": False}, done)
    assert cap_block("tune_arima", done) is not None

def test_cached_and_blocked_do_not_count():
    done = {}
    count_call("make_report", OK, {"ok": True, "cached": True}, done)
    count_call("make_report", {"ok": False}, {"ok": False, "blocked": True}, done)
    assert cap_block("make_report", done) is None

def test_call_cap_is_a_loop_block_in_metrics():
    assert "CALL_CAP" in LOOP_BLOCKS
