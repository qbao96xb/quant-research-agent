"""One-way boundary: coders edit the quant agent; the agent never touches coder files or source.

The agent acts only through registered tools. Tools write only under data/, artifacts/,
results/ and logs/, and product code never refers to the coders' files.
"""
from pathlib import Path

import pytest

import src.agent.findings as findings
import src.agent.logging_utils as logging_utils
import src.agent.session as session
import src.backtest.jobs as jobs
import src.data.loader as loader
import src.report.pipeline as pipeline
import src.tools.cache as cache
import src.tools.estimate as estimate
from src.tools.registry import call

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIRS = [ROOT / d for d in ("data", "artifacts", "results", "logs")]
CODER_NAMES = ("handoff/", ".claude/", "CLAUDE.md", "CONTRACT.md")

WRITE_PATHS = {
    "cache.STORE": cache.STORE,
    "jobs.JOBS": jobs.JOBS,
    "session.SESS_DIR": session.SESS_DIR,
    "logging_utils.LOGS": logging_utils.LOGS,
    "pipeline.REPORTS_DIR": pipeline.REPORTS_DIR,
    "loader.CACHE": loader.CACHE,
    "loader.SNAPSHOTS": loader.SNAPSHOTS,
    "estimate.CALIB": estimate.CALIB,
    "findings.chat": findings.ROOT / "results" / "chat",
}


@pytest.mark.parametrize("name", sorted(WRITE_PATHS))
def test_write_paths_stay_in_output_dirs(name):
    p = WRITE_PATHS[name].resolve()
    assert any(p.is_relative_to(d) for d in OUTPUT_DIRS), f"{name} -> {p}"


def test_product_code_never_mentions_coder_files():
    src = ROOT / "src"
    hits = []
    for p in sorted(src.rglob("*")):
        if p.suffix not in (".py", ".md") or "tests" in p.relative_to(src).parts:
            continue
        text = p.read_text(errors="ignore")
        hits += [f"{p.relative_to(ROOT)}: {n}" for n in CODER_NAMES if n in text]
    assert hits == []


@pytest.mark.parametrize("tool", ["load_data", "make_report"])
@pytest.mark.parametrize("ticker", ["../../src", "KO/../../x", "/tmp/x", ".claude"])
def test_path_like_tickers_are_rejected(tool, ticker, monkeypatch):
    def never(*a, **k):
        raise AssertionError("report must not be built for a bad ticker")
    monkeypatch.setattr(pipeline, "make_report", never)
    r = call(tool, ticker=ticker)
    assert not r["ok"] and r["error"]["code"] == "BAD_TICKER"
