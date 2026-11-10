import pytest
import src.report.pipeline as pl

def test_report_no_llm(facts, tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "collect_facts", lambda t, **kw: facts)
    monkeypatch.setattr(pl, "ROOT", tmp_path)
    monkeypatch.setattr(pl, "log_event", lambda **kw: None)
    md = pl.make_report("KO", llm=None)
    out = md.parent
    assert md.exists() and (out / "facts.json").exists()
    assert len(list((out / "figures").glob("*.png"))) >= 4
    text = md.read_text()
    for n in range(1, 9):
        assert f"## {n}." in text
    assert "narrative: fallback" in text

def test_tool_failure_leaves_no_folder(tmp_path, monkeypatch):
    def boom(t, **kw): raise RuntimeError("load_data failed: X")
    monkeypatch.setattr(pl, "collect_facts", boom)
    monkeypatch.setattr(pl, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="load_data failed"):
        pl.make_report("BAD")
    assert not (tmp_path / "reports").exists()