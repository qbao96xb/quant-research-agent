"""A turn that only repeats successful calls ends the run with a forced final answer (S03, owner option A, 2026-11-16)."""
import json

import pytest

import src.agent.loop as loop
from src.agent.repeats import only_repeats

OK = {"ok": True, "mean_pct": "+0.28%"}


def tool_call(name, args=None, cid="c1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


def reply(content="", calls=None):
    m = {"role": "assistant", "content": content}
    if calls:
        m["tool_calls"] = calls
    return m


class RepeatLLM:
    """Calls forecast, then repeats it; answers only when tools is None (forced final call)."""
    def __init__(self):
        self.n = 0
        self.tools_seen = []

    def chat(self, messages, tools=None):
        self.n += 1
        self.tools_seen.append(tools is not None)
        if tools is None:
            return reply("KO forecast +0.28%")
        return reply(calls=[tool_call("forecast", {"h": 5}, cid="c" + str(self.n))])


@pytest.fixture

def made(monkeypatch):
    schema = [{"type": "function", "function": {
        "name": "forecast", "description": "x",
        "parameters": {"type": "object", "properties": {}}}}]
    monkeypatch.setattr(loop, "schemas", lambda: schema)
    monkeypatch.setattr(loop, "build_system_prompt", lambda: "SYSTEM")
    monkeypatch.setattr(loop, "log_event", lambda **kw: None)
    names = []

    def fake(name, args):
        names.append(name)
        return OK, {"ok": True}
    monkeypatch.setattr(loop, "safe_call", fake)
    return names


def test_repeat_turn_forces_final_answer(made):
    llm = RepeatLLM()
    r = loop.run_agent("Forecast KO for 5 days", llm)
    assert made == ["forecast"]
    assert r["status"] == "done"
    assert r["steps"] == 2
    assert r["answer"] == "KO forecast +0.28%"
    assert llm.tools_seen[:3] == [True, True, False]
    assert len([m for m in r["messages"] if m["role"] == "tool"]) == 1


def test_only_repeats_needs_every_call_cached():
    key = loop._key
    cache = {key("forecast", '{"h": 5}'): (OK, {"ok": True})}
    same = [tool_call("forecast", {"h": 5})]
    mixed = same + [tool_call("forecast", {"h": 10}, cid="c2")]
    log = lambda **kw: None
    assert only_repeats(same, cache, key, log, "r1", 2) is True
    assert only_repeats(mixed, cache, key, log, "r1", 2) is False
    assert only_repeats([], cache, key, log, "r1", 2) is False
