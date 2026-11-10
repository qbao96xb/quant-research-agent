import json
from pathlib import Path

import src.report.pipeline as pl
from src.tools import report_tools as rt
from src.tools.registry import call
from src.agent.findings import write_findings


def _fake(tmp_path, monkeypatch):
    out = tmp_path / "rep"; (out / "figures").mkdir(parents=True)
    (out / "figures" / "a.png").write_bytes(b"x")
    (out / "report.md").write_text("*Chart unavailable: MISSING (boom)*\n")
    monkeypatch.setattr(rt, "ROOT", tmp_path)
    monkeypatch.setattr(pl, "make_report", lambda t, llm=None, **kw: out / "report.md")
    return out


def test_make_report_tool(tmp_path, monkeypatch):
    _fake(tmp_path, monkeypatch)
    r = call("make_report", ticker="ko", h=5)
    assert r["ok"] and r["ticker"] == "KO" and "naive" in r["models"]
    assert r["figures"] == ["rep/figures/a.png"] and len(r["charts_failed"]) == 1


def test_make_report_failure_is_tool_error(monkeypatch):
    def boom(*a, **k): raise RuntimeError("x")
    monkeypatch.setattr(pl, "make_report", boom)
    assert call("make_report", ticker="KO")["error"]["code"] == "REPORT_FAILED"


def test_findings_gallery_includes_report_figures(tmp_path, monkeypatch):
    import src.agent.findings as fd
    (tmp_path / "figures").mkdir(); (tmp_path / "figures" / "a.png").write_bytes(b"x")
    monkeypatch.setattr(fd, "ROOT", tmp_path)
    msgs = [{"role": "user", "content": "report"},
            {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "make_report", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": json.dumps({"ok": True, "figures": ["figures/a.png"]})}]
    _, _, charts = write_findings(dict(run_id="t1", status="ok", answer="done", messages=msgs), "report")
    assert charts == [tmp_path / "figures" / "a.png"]
