"""A failed load_data must not be replaced by saved data (eval case I01)."""
import json

import pytest

import src.agent.loop as loop

OK = {"ok": True, "data": {"n": 1}}


def tool_call(name, args=None, cid="c1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


def reply(content="", calls=None):
    m = {"role": "assistant", "content": content}
    if calls:
        m["tool_calls"] = calls
    return m


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.n = 0

    def chat(self, messages, tools=None):
        r = self.replies[min(self.n, len(self.replies) - 1)]
        self.n += 1
        return r


@pytest.fixture
def calls(monkeypatch):
    schema = [{"type": "function", "function": {
        "name": "load_data", "description": "x",
        "parameters": {"type": "object", "properties": {}}}}]
    monkeypatch.setattr(loop, "schemas", lambda: schema)
    monkeypatch.setattr(loop, "build_system_prompt", lambda: "SYSTEM")
    monkeypatch.setattr(loop, "log_event", lambda **kw: None)
    return []


def use_tools(monkeypatch, calls, load_result):
    def fake(name, args):
        calls.append(name)
        r = load_result if name == "load_data" else OK
        return r, {"ok": bool(r.get("ok"))}
    monkeypatch.setattr(loop, "safe_call", fake)


def script(first):
    return ScriptedLLM([reply(calls=first),
                        reply(calls=[tool_call("list_artifacts", cid="c2")]),
                        reply("XXXX could not be loaded. Please check the symbol.")])


@pytest.mark.parametrize("code", ["DATA_FETCH_FAILED", "BAD_TICKER"])
def test_failed_load_blocks_list_artifacts(monkeypatch, calls, code):
    fail = {"ok": False, "error": {"code": code, "message": "No data returned for XXXX"}}
    use_tools(monkeypatch, calls, fail)
    r = loop.run_agent("Forecast XXXX for 5 days",
                       script([tool_call("load_data", {"ticker": "XXXX"})]))
    assert calls == ["load_data"]
    tool_msgs = [m["content"] for m in r["messages"] if m["role"] == "tool"]
    assert "NO_FALLBACK" in tool_msgs[1]
    assert r["status"] == "done"


def test_list_artifacts_runs_after_successful_load(monkeypatch, calls):
    use_tools(monkeypatch, calls, OK)
    loop.run_agent("Forecast KO", script([tool_call("load_data", {"ticker": "KO"})]))
    assert calls == ["load_data", "list_artifacts"]


def test_list_artifacts_runs_without_load(monkeypatch, calls):
    use_tools(monkeypatch, calls, OK)
    llm = ScriptedLLM([reply(calls=[tool_call("list_artifacts")]), reply("done")])
    loop.run_agent("What have I saved?", llm)
    assert calls == ["list_artifacts"]
