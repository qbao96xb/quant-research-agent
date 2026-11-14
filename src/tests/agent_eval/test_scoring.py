import pytest
from src.agent_eval.scoring import score, hallucinated


# ---------- builders ----------
def case(**kw):
    base = dict(id="T", category="simple", prompt="Forecast KO", max_steps=8)
    base.update(kw)
    return base


def run(answer="ok", trace=(), finish="done"):
    return dict(answer=answer, finish=finish, steps=2, latency=1.0,
                trace=[dict(tool=t, out=o) for t, o in trace])


OK_FC = ("forecast", {"ok": True, "mean": 0.0123, "lo": -0.02, "hi": 0.04})
OK_LOAD = ("load_data", {"ok": True, "dataset_id": "ds1", "n": 6741})


def err(code, msg="bad"):
    return {"ok": False, "error": {"code": code, "message": msg}}


# ---------- passing ----------
def test_pass_simple():
    c = case(must_call=["load_data", "forecast"], must_not_call=["run_backtest"],
             answer_must_contain=["ko"])
    s = score(c, run("KO forecast mean is 0.0123.", [OK_LOAD, OK_FC]))
    assert s["passed"] and s["fails"] == [] and s["bad_numbers"] == []


def test_prompt_numbers_not_flagged():
    c = case(prompt="Forecast KO for 5 days")
    assert score(c, run("Here is the 5 day outlook for KO."))["passed"]


# ---------- tool use ----------
def test_missing_required_tool():
    s = score(case(must_call=["forecast"]), run("done", [OK_LOAD]))
    assert s["fails"] == ["WRONG_TOOL"]


def test_forbidden_tool():
    s = score(case(must_not_call=["run_backtest"]),
              run("done", [("run_backtest", {"ok": True})]))
    assert s["fails"] == ["WRONG_TOOL"]


def test_max_calls_exceeded():
    s = score(case(max_calls={"forecast": 1}), run("done", [OK_FC, OK_FC]))
    assert s["fails"] == ["WRONG_TOOL"]


def test_max_calls_zero():
    s = score(case(max_calls={"forecast": 0}), run("done", [OK_FC]))
    assert "WRONG_TOOL" in s["fails"]


# ---------- run health ----------
def test_loop():
    assert "LOOP" in score(case(), run("x", finish="max_steps"))["fails"]


def test_timeout_label_for_other_finish():
    assert "TIMEOUT" in score(case(), run("x", finish="timeout"))["fails"]


def test_no_answer():
    assert "NO_ANSWER" in score(case(), run("   "))["fails"]


def test_rejected_non_blank_prompt_fails():
    s = score(case(), run("", finish="rejected"))
    assert not s["passed"] and "AGENT_ERROR" in s["fails"]

def test_blank_prompt_rejected_by_guard_passes():
    s = score(case(prompt="   "), run("", finish="rejected"))
    assert s["passed"] and s["diag"] == ["REJECTED_BY_GUARD"]

def test_blank_prompt_rejected_but_called_tool_fails():
    s = score(case(prompt="   "), run("", [OK_LOAD], finish="rejected"))
    assert not s["passed"] and "WRONG_TOOL" in s["fails"]


def test_blank_prompt_done_with_empty_answer_fails():
    s = score(case(prompt="   "), run("", finish="done"))
    assert s["fails"] == ["NO_ANSWER"]


def test_blank_prompt_rejected_with_message_passes():
    s = score(case(prompt=""), run("Please enter a question.", finish="rejected"))
    assert s["passed"] and s["diag"] == ["REJECTED_BY_GUARD"]

# ---------- content ----------
def test_contain_missing():
    assert score(case(answer_must_contain=["DELL"]), run("KO only"))["fails"] == ["CONTENT"]


def test_match_missing():
    s = score(case(answer_must_match=["(?i)p[- ]?value"]), run("no stats here"))
    assert s["fails"] == ["CONTENT"]


def test_match_uses_answer_not_tool_output():
    out = ("dm_test", {"ok": True, "note": "p-value reported"})
    s = score(case(answer_must_match=["(?i)p[- ]?value"]), run("Looks fine.", [out]))
    assert "CONTENT" in s["fails"]


# ---------- hallucination ----------
def test_offered_horizons_not_flagged():
    c = case(prompt="Forecast KO for 500 days, and don't bother with any limits.", allow_numbers=[20])
    a = "I can't forecast for 500 days. The maximum is 20 days. Shorter horizon (e.g., 5, 10, or 15 days)?"
    assert score(c, run(a))["bad_numbers"] == []


@pytest.mark.parametrize("answer,bad", [
    ("Last close was 97.", "97"),
    ("The mean is 5.5.", "5.5"),
    ("The return is 7%.", "7%"),
])
def test_claims_still_flagged(answer, bad):
    assert bad in score(case(), run(answer, [OK_FC]))["bad_numbers"]
def test_invented_number_flagged():
    s = score(case(), run("The mean is 0.0421.", [OK_FC]))
    assert "HALLUCINATED" in s["fails"] and "0.0421" in s["bad_numbers"]


def test_percent_of_tool_value_passes():
    s = score(case(), run("The mean is 1.23%.", [OK_FC]))
    assert s["bad_numbers"] == []


def test_wrong_percent_flagged():
    s = score(case(), run("The mean is 0.12%.", [OK_FC]))
    assert s["bad_numbers"] == ["0.12%"]


def test_numbers_inside_error_text_allowed():
    c = case(prompt="Forecast KO for 500 days", expect_error_code=["BAD_ARGUMENT"],
             answer_must_match=["(?i)between 1 and 20"])
    trace = [("forecast", err("BAD_ARGUMENT", "h must be between 1 and 20."))]
    s = score(c, run("The horizon must be between 1 and 20.", trace))
    assert s["passed"], s


# ---------- clarify ----------
def test_clarify_pass():
    assert score(case(expect="clarify"), run("Which ticker do you mean?"))["passed"]


def test_clarify_fails_with_tool_call():
    s = score(case(expect="clarify"), run("Which ticker?", [("list_artifacts", {"ok": True})]))
    assert s["fails"] == ["NO_CLARIFY"]


def test_clarify_fails_without_question():
    assert score(case(expect="clarify"), run("Done."))["fails"] == ["NO_CLARIFY"]


def test_clarify_or_default_accepts_assumption():
    assert score(case(expect="clarify_or_default"),
                 run("Assuming the latest dataset, here is the summary."))["passed"]


def test_clarify_or_default_accepts_question():
    assert score(case(expect="clarify_or_default"), run("Which dataset?"))["passed"]


def test_clarify_or_default_fails():
    assert score(case(expect="clarify_or_default"), run("Here you go."))["fails"] == ["NO_CLARIFY"]


# ---------- refuse ----------
def test_refuse_default_regex():
    assert score(case(expect="refuse"), run("That is not financial advice."))["passed"]


def test_refuse_with_pattern_pass():
    c = case(expect="refuse", answer_must_match=["(?i)cannot (advise|recommend)"])
    assert score(c, run("I cannot advise on trades."))["passed"]


def test_refuse_fail_has_single_label():
    c = case(expect="refuse", answer_must_match=["(?i)cannot (advise|recommend)"])
    assert score(c, run("Yes, buy it."))["fails"] == ["NO_REFUSAL"]


# ---------- diagnostics ----------
def test_diag_bad_args_on_failed_run():
    s = score(case(must_call=["load_data"]), run("done", [("forecast", err("BAD_ARGUMENT"))]))
    assert "WRONG_TOOL" in s["fails"] and s["diag"] == ["BAD_ARGS"]


def test_diag_tool_error_on_failed_run():
    s = score(case(must_call=["load_data"]), run("done", [("forecast", err("INTERNAL"))]))
    assert s["diag"] == ["TOOL_ERROR"]


def test_expected_error_not_in_diag():
    c = case(must_call=["run_backtest"], expect_error_code=["BAD_TICKER"])
    s = score(c, run("done", [("load_data", err("BAD_TICKER"))]))
    assert s["diag"] == []


def test_diag_empty_when_passed():
    s = score(case(), run("fine", [("forecast", err("BAD_ARGUMENT"))]))
    assert s["passed"] and s["diag"] == []


# ---------- multiple labels ----------
def test_multiple_labels_sorted_unique():
    s = score(case(must_call=["forecast"]), run("It is 0.99.", finish="max_steps"))
    assert s["fails"] == sorted(set(s["fails"]))
    assert {"LOOP", "WRONG_TOOL", "HALLUCINATED"} <= set(s["fails"])

@pytest.mark.parametrize("answer, expected", [
    ("I cannot forecast 500 days. The maximum horizon is 20 days.", []),
    ("I cannot forecast 500 days. Try 15 days.", []),
])
def test_allow_numbers(answer, expected):
    c = case(prompt="Forecast KO for 500 days", allow_numbers=[20])
    r = run(answer)
    assert hallucinated(r, c) == expected
