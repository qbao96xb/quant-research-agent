"""The loop runs a capped tool at most once per request (src/agent/scope.py, owner 2026-11-13)."""
import json

import pytest

import src.agent.loop as loop

OK = {"ok": True, "data": {"n": 1}}
BAD = {"ok": False, "error": {"code": "BAD_ARGS", "message": "bad"}}


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
        "name": "make_report", "description": "x",
        "parameters": {"type": "object", "properties": {}}}}]
    monkeypatch.setattr(loop, "schemas", lambda: schema)
    monkeypatch.setattr(loop, "build_system_prompt", lambda: "SYSTEM")
    monkeypatch.setattr(loop, "log_event", lambda **kw: None)
    return []


def use_tools(monkeypatch, calls, results):
    def fake(name, args):
        calls.append(name)
        r = results.pop(0) if results else OK
        return r, {"ok": bool(r.get("ok"))}
    monkeypatch.setattr(loop, "safe_call", fake)


def two_reports(first, second):
    return ScriptedLLM([reply(calls=[tool_call("make_report", first, "c1")]),
                        reply(calls=[tool_call("make_report", second, "c2")]),
                        reply("Report done.")])


def tool_msgs(r):
    return [m["content"] for m in r["messages"] if m["role"] == "tool"]


def test_second_capped_call_is_blocked(monkeypatch, calls):
    use_tools(monkeypatch, calls, [OK])
    r = loop.run_agent("Report on KO", two_reports({"ticker": "KO"}, {"ticker": "KO", "h": 10}))
    assert calls == ["make_report"]
    assert "CALL_CAP" in tool_msgs(r)[1]
    assert r["status"] == "done"


def test_identical_repeat_ends_with_answer(monkeypatch, calls):
    use_tools(monkeypatch, calls, [OK])
    r = loop.run_agent("Report on KO", two_reports({"ticker": "KO"}, {"ticker": "KO"}))
    assert calls == ["make_report"]
    assert len(tool_msgs(r)) == 1
    assert r["answer"] == "Report done."


def test_failed_call_does_not_count(monkeypatch, calls):
    use_tools(monkeypatch, calls, [BAD, OK])
    r = loop.run_agent("Report on KO", two_reports({"ticker": "KO", "h": -1}, {"ticker": "KO"}))
    assert calls == ["make_report", "make_report"]
    assert "CALL_CAP" not in tool_msgs(r)[1]


def test_new_request_resets_the_cap(monkeypatch, calls):
    use_tools(monkeypatch, calls, [OK, OK])
    loop.run_agent("Report on KO", ScriptedLLM([reply(calls=[tool_call("make_report", {"ticker": "KO"})]),
                                                reply("Report done.")]))
    loop.run_agent("Report on KO for 10 days", ScriptedLLM([reply(calls=[tool_call("make_report", {"ticker": "KO", "h": 10})]),
                                                            reply("Report done.")]))
    assert calls == ["make_report", "make_report"]
