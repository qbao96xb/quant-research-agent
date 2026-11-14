import json
from pathlib import Path

import pytest

import src.agent.loop as loop

SYSTEM_MD = Path(__file__).parents[2] / "agent" / "prompts" / "system.md"

LOAD_OUT = {"ok": True, "dataset_id": "ds_xom", "n_rows": 2500}
TUNE_OUT = {
    "ok": True, "tune_id": "tune_abc", "best_order": [2, 0, 1],
    "held_out_check": "pass",
    "check_reason": "Best order is not significantly worse than naive or ARIMA(1,0,1) on the held-out test.",
    "orders_tried": 9, "n_val": 60, "n_test": 40,
    "test_rmse": {"best": 0.0312, "default_101": 0.0318, "naive": 0.0325},
    "dm_best_vs_naive": {"p": 0.41, "dm": -0.83, "verdict": "no difference", "n": 40},
    "dm_best_vs_default": {"p": 0.57, "dm": -0.57, "verdict": "no difference", "n": 40},
}
FC_OUT = {"ok": True, "pred_return_pct": 0.84, "lo90_pct": -3.12, "hi90_pct": 4.67}
OUTS = {"load_data": LOAD_OUT, "tune_arima": TUNE_OUT, "forecast": FC_OUT}
CALLS = []


def call(name, args, cid):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


class ScriptedLLM:
    def __init__(self, replies):
        self.replies, self.n = list(replies), 0

    def chat(self, messages, tools=None):
        r = self.replies[min(self.n, len(self.replies) - 1)]
        self.n += 1
        return r



def fake_call(name, args):
    CALLS.append((name, args))
    return OUTS[name], {"ok": True}


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    CALLS.clear()
    schema = [{"type": "function", "function": {
        "name": n, "description": "x",
        "parameters": {"type": "object", "properties": {}}}} for n in OUTS]
    monkeypatch.setattr(loop, "schemas", lambda: schema)
    monkeypatch.setattr(loop, "build_system_prompt", lambda: "SYSTEM")
    monkeypatch.setattr(loop, "log_event", lambda **kw: None)
    monkeypatch.setattr(loop, "safe_call", fake_call)



def script(answer):
    return ScriptedLLM([
        call("load_data", {"ticker": "XOM"}, "c1"),
        call("tune_arima", {"dataset_id": "ds_xom", "h": 5}, "c2"),
        call("forecast", {"dataset_id": "ds_xom", "model": "arima", "h": 5,
                          "order": [2, 0, 1]}, "c3"),
        {"role": "assistant", "content": answer},
    ])


GOOD = ("Best order (2, 0, 1) from 9 orders (n_val 60, n_test 40). Held-out check: pass. "
        "DM p-value vs naive 0.41 and vs ARIMA(1,0,1) 0.57: no significant difference. "
        "5-day ARIMA forecast: 0.84% (90% interval -3.12% to 4.67%).")



def test_pass_path_forecasts_with_best_order():
    r = loop.run_agent("Tune ARIMA for XOM and forecast", script(GOOD))
    assert r["status"] == "done"
    assert [c[0] for c in CALLS] == ["load_data", "tune_arima", "forecast"]
    fc_args = CALLS[2][1]
    if isinstance(fc_args, str):
        fc_args = json.loads(fc_args)
    assert fc_args["order"] == TUNE_OUT["best_order"]
    assert fc_args["dataset_id"] == LOAD_OUT["dataset_id"]
    assert r["unverified"] == []



def test_invented_number_is_flagged():
    bad = GOOD.replace("0.84%", "0.99%")
    r = loop.run_agent("Tune ARIMA for XOM and forecast", script(bad))
    assert "0.99%" in r["unverified"]



def test_prompt_has_pass_rule():
    text = SYSTEM_MD.read_text()
    tuning = text.split("# Tuning")[1].split("\n# ")[0]
    assert 'If `held_out_check` is "pass"' in tuning
    assert "order=best_order" in tuning
    assert 'If `held_out_check` is "fail"' in tuning
    assert "Never choose or compare orders yourself" in tuning


TUNE_FAIL = {
    **TUNE_OUT, "best_order": [2, 0, 2], "held_out_check": "fail",
    "check_reason": ("Best order is significantly worse on the held-out test than "
                     "naive (DM p=0.013) and ARIMA(1,0,1) (DM p=0.015)."),
    "dm_best_vs_naive": {"p": 0.013, "dm": 2.6, "verdict": "A worse", "n": 40},
    "dm_best_vs_default": {"p": 0.015, "dm": 2.5, "verdict": "A worse", "n": 40},
}
FC_101 = {"ok": True, "pred_return_pct": 0.52, "lo90_pct": -3.4, "hi90_pct": 4.41}
FAIL_Q = ("Best order (2, 0, 2) from 9 orders (n_val 60, n_test 40). Held-out check: fail. "
          "Test RMSE 0.0312 (best), 0.0318 (1,0,1), 0.0325 (naive). "
          "DM p-value vs naive 0.013, vs ARIMA(1,0,1) 0.015. "
          "Which order should I use: 1) keep (2, 0, 2), 2) fall back to the default (1, 0, 1), "
          "3) re-tune on a different split?")
FALLBACK = ("Applied option 2: ARIMA(1, 0, 1) on the same dataset, because the tuned order "
             "failed the held-out check (DM p-value vs naive 0.013). "
             "5-day forecast: 0.52% (90% interval -3.40% to 4.41%).")


def test_fail_path_asks_then_falls_back(monkeypatch):
    monkeypatch.setitem(OUTS, "tune_arima", TUNE_FAIL)
    t1 = ScriptedLLM([
        call("load_data", {"ticker": "XOM"}, "c1"),
        call("tune_arima", {"dataset_id": "ds_xom", "h": 5}, "c2"),
        {"role": "assistant", "content": FAIL_Q},
    ])
    r1 = loop.run_agent("Tune ARIMA for XOM and forecast", t1)
    assert r1["status"] == "done"
    assert [c[0] for c in CALLS] == ["load_data", "tune_arima"]
    assert r1["unverified"] == []
    assert r1["answer"].count("?") == 1

    monkeypatch.setitem(OUTS, "forecast", FC_101)
    CALLS.clear()
    t2 = ScriptedLLM([
        call("forecast", {"dataset_id": "ds_xom", "model": "arima", "h": 5,
                      "order": [1, 0, 1]}, "c3"),
        {"role": "assistant", "content": FALLBACK},
    ])
    r2 = loop.run_agent("2, fall back to the default", t2, history=r1["history"])
    assert r2["status"] == "done"
    assert [c[0] for c in CALLS] == ["forecast"]
    fc_args = CALLS[0][1]
    if isinstance(fc_args, str):
        fc_args = json.loads(fc_args)
    assert fc_args["order"] == [1, 0, 1]
    assert fc_args["dataset_id"] == "ds_xom"
    assert any(m["role"] == "tool" and "ds_xom" in m["content"] for m in r1["history"])
    assert r2["unverified"] == []


def test_prompt_has_fail_rule():
    text = SYSTEM_MD.read_text()
    tuning = text.split("# Tuning")[1].split("\n# ")[0]
    assert "exactly one question" in tuning
    assert "fall back to the default (1,0,1)" in tuning
    assert "do not call `tune_arima` again" in tuning
    assert "Never choose or compare orders yourself" in tuning

def test_prompt_has_retune_rule():
    text = SYSTEM_MD.read_text()
    tuning = text.split("# Tuning")[1].split("\n# ")[0]
    assert "For option 3: call `tune_arima` once" in tuning
    assert "quote the re-tune warning" in tuning
    assert "Never re-tune on your own" in tuning
    assert len([x for x in tuning.splitlines() if x.startswith("- ")]) == 10
    assert "Never choose or compare orders yourself" in tuning
