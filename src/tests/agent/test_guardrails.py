import json
from src.agent.guardrails import safe_call, truncate


def test_truncate_short_text_unchanged():
    assert truncate("abc") == "abc"


def test_truncate_long_text_is_shorter():
    s = "x" * 100_000
    assert len(truncate(s)) < len(s)


def test_bad_json_arguments():
    result, meta = safe_call("load_data", "{not json")
    assert result["ok"] is False


def test_unknown_tool():
    result, meta = safe_call("does_not_exist", "{}")
    assert result["ok"] is False


def test_empty_arguments_do_not_crash():
    result, meta = safe_call("list_artifacts", "")
    assert isinstance(result, dict)
    assert isinstance(meta, dict)
