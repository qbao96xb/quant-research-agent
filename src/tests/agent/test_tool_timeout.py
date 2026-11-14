import inspect

import src.agent.guardrails as guardrails
import src.agent.loop as loop


def test_safe_call_default_timeout_matches_loop():
    default = inspect.signature(guardrails.safe_call).parameters["timeout"].default
    assert default == loop.TOOL_TIMEOUT


def test_loop_call_gets_full_tool_timeout(monkeypatch):
    seen = {}

    def fake_run(name, args, timeout):
        seen["timeout"] = timeout
        return {"ok": True}

    monkeypatch.setattr(guardrails, "_run", fake_run)
    monkeypatch.setattr(guardrails, "REGISTRY", {"slow_tool": None})
    guardrails.safe_call("slow_tool", "{}")
    assert seen["timeout"] == loop.TOOL_TIMEOUT
