import json
import itertools
import pytest

import src.agent.loop as loop


def tool_call(name, args=None, cid="c1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


def reply(content="", calls=None):
    m = {"role": "assistant", "content": content}
    if calls:
        m["tool_calls"] = calls
    return m


class ScriptedLLM:
    """Returns scripted replies in order. Repeats the last one if exhausted."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.n = 0

    def chat(self, messages, tools=None):
        r = self.replies[min(self.n, len(self.replies) - 1)]
        self.n += 1
        return r

class EndlessLLM:
    """Never answers; every reply is a distinct tool call, so the dedupe guard never fires."""
    def __init__(self):
        self.n = 0

    def chat(self, messages, tools=None):
        self.n += 1
        return reply(calls=[tool_call("load_data", {"ticker": f"T{self.n}"}, cid=f"c{self.n}")])

class DupLLM:
    """Repeats the same call; returns a final answer only when tools is None (forced final call)."""
    def __init__(self):
        self.n = 0

    def chat(self, messages, tools=None):
        self.n += 1
        if tools is None:
            return reply("final from cache")
        return reply(calls=[tool_call("load_data", {"ticker": "KO"}, cid=f"c{self.n}")])


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    schema = [{"type": "function", "function": {
        "name": "load_data", "description": "x",
        "parameters": {"type": "object", "properties": {}}}}]
    monkeypatch.setattr(loop, "schemas", lambda: schema)
    monkeypatch.setattr(loop, "build_system_prompt", lambda: "SYSTEM")
    monkeypatch.setattr(loop, "log_event", lambda **kw: None)
    monkeypatch.setattr(
        loop, "safe_call",
        lambda name, args: ({"ok": True, "data": {"n": 1}}, {"ok": True}))


def test_answer_without_tools():
    r = loop.run_agent("hi", ScriptedLLM([reply("hello")]))
    assert r["status"] == "done"
    assert r["answer"] == "hello"
    assert r["steps"] == 1


def test_tool_then_answer():
    llm = ScriptedLLM([reply(calls=[tool_call("load_data", {"ticker": "KO"})]),
                       reply("final")])
    r = loop.run_agent("load KO", llm)
    assert r["status"] == "done"
    assert r["steps"] == 2
    tool_msgs = [m for m in r["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "c1"


def test_multiple_calls_in_one_reply():
    calls = [tool_call("load_data", cid="a"), tool_call("load_data", cid="b")]
    llm = ScriptedLLM([reply(calls=calls), reply("ok")])
    r = loop.run_agent("x", llm)
    ids = [m["tool_call_id"] for m in r["messages"] if m["role"] == "tool"]
    assert ids == ["a", "b"]

def test_max_steps():
    llm = EndlessLLM()
    r = loop.run_agent("x", llm, max_steps=3)
    assert r["status"] == "max_steps"
    assert r["steps"] == 3
    assert llm.n == 4          # 3 loop steps + 1 forced no-tools summary call
    assert r["answer"]  # never empty

def test_timeout():
    llm = ScriptedLLM([reply("never reached")])
    r = loop.run_agent("x", llm, run_timeout=-1)
    assert r["status"] == "timeout"
    assert llm.n == 0


def test_bad_ticker_is_passed_to_model(monkeypatch):
    err = {"ok": False, "error": {"code": "BAD_TICKER", "message": "no data"}}
    monkeypatch.setattr(loop, "safe_call",
                        lambda name, args: (err, {"ok": False}))
    llm = ScriptedLLM([reply(calls=[tool_call("load_data", {"ticker": "XXXX"})]),
                       reply("BAD_TICKER: XXXX has no data.")])
    r = loop.run_agent("Forecast XXXX", llm)
    tool_msg = next(m for m in r["messages"] if m["role"] == "tool")
    assert "BAD_TICKER" in tool_msg["content"]
    assert r["status"] == "done"
    assert "BAD_TICKER" in r["answer"]

def test_duplicate_calls_force_final_answer():
    r = loop.run_agent("x", DupLLM(), max_steps=10)
    assert r["status"] == "done"
    assert r["steps"] == 3
    assert r["answer"] == "final from cache"
    notes = [m for m in r["messages"]
             if m["role"] == "tool" and "Duplicate call" in m["content"]]
    assert len(notes) == 2

def test_running_status_not_cached(monkeypatch):
    seq = iter([{"ok": True, "status": "running"},
                {"ok": True, "status": "running"},
                {"ok": True, "status": "done"}])
    monkeypatch.setattr("agent.loop.safe_call",
                        lambda n, a: (next(seq), {"ok": True}))
    # drive run_agent with a fake llm that calls backtest_status three times
    # assert no tool message contains "Duplicate call"
