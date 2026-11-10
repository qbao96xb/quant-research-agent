# src/report/pipeline.py
import json
import shutil
from datetime import date
from pathlib import Path

from .facts import collect_facts
from .charts import build_all
from .narrative import write_narrative, fallback
from .builder import build_markdown
from src.agent.logging_utils import log_event

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "results" / "reports"


def _relative(figs, out):
    """Make figure paths relative to the report folder, so links survive a move."""
    def rel(p):
        if isinstance(p, (list, tuple)):
            return [rel(q) for q in p]
        s = str(p)
        if s.startswith("MISSING"):
            return s
        try:
            return str(Path(s).resolve().relative_to(out.resolve()))
        except ValueError:
            return s
    return {k: rel(v) for k, v in figs.items()}


def _load_preds(f):
    """Per-origin backtest rows (None if unavailable, so only those charts are skipped)."""
    try:
        from src.tools.cache import get
        return get(f["backtest_id"], "bt").assign(ticker=f["ticker"])
    except Exception:
        return None


def _narrative(f, llm):
    """LLM narrative if available, otherwise (or on any LLM error) the fallback."""
    if llm is None:
        return fallback(f)
    try:
        return write_narrative(f, llm)
    except Exception as e:
        log_event(event="narrative_error", ticker=f["ticker"], error=repr(e))
        return fallback(f)


def make_report(ticker, llm=None, **kw):
    """Build results/reports/<TICKER>_<YYYYMMDD>/report.md and return its path.

    **kw is forwarded to collect_facts (start, h, models, bt_start, ...).
    The report is built in a temp folder and moved into place only on success,
    so a failed run never leaves a partial or mixed folder.
    """
    try:
        f = collect_facts(ticker, **kw)
    except Exception as e:
        log_event(event="report_failed", ticker=ticker, stage="facts", error=repr(e))
        raise

    final = REPORTS_DIR / f"{ticker}_{date.today():%Y%m%d}"
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    try:
        (tmp / "facts.json").write_text(json.dumps(f, indent=2, default=str), encoding="utf-8")
        preds = _load_preds(f)
        if preds is not None:
            preds.to_csv(tmp / f"{f['ticker']}_preds.csv", index=False)
        figs = _relative(build_all(f, tmp, preds=preds), tmp)
        text = _narrative(f, llm)
        (tmp / "report.md").write_text(build_markdown(f, figs, text), encoding="utf-8")
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        log_event(event="report_failed", ticker=ticker, stage="build", error=repr(e))
        raise

    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    log_event(event="report", ticker=ticker, path=str(final),
              narrative=text.get("status", "unknown"))
    return final / "report.md"
