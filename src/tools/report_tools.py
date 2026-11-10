# src/tools/report_tools.py
from pathlib import Path

from .errors import ToolError
from .registry_core import tool
from .schemas import ReportArgs

ROOT = Path(__file__).resolve().parents[2]


@tool(ReportArgs, slow=True)
def make_report(args) -> dict:
    """Build a full research report for one ticker: forecast fan chart, backtest charts,
    calibration (PIT, Mincer-Zarnowitz, error distribution) and GARCH volatility diagnostics.

    Loads the data and runs the backtest itself, so do NOT call load_data or run_backtest first.
    Returns `report` (path to report.md) and `figures` (PNG paths). Quote the paths as written.
    `charts_failed` lists charts that could not be drawn; mention them if not empty.
    """
    from src.report.pipeline import make_report as build   # lazy: avoids a circular import
    models = list(dict.fromkeys(["naive", *args.models]))
    try:
        md = build(args.ticker.upper(), llm=None, h=args.h, models=tuple(models), start=args.start)
    except Exception as e:
        raise ToolError("REPORT_FAILED", f"{type(e).__name__}: {e}",
                        "Check the ticker, or try a smaller model list.")
    rel = lambda p: str(Path(p).relative_to(ROOT))
    figures = sorted((md.parent / "figures").glob("*.png"))
    failed = [l.strip("* ") for l in md.read_text().splitlines() if "Chart unavailable" in l]
    return dict(ok=True, ticker=args.ticker.upper(), h=args.h, models=models,
                report=rel(md), figures=[rel(p) for p in figures], charts_failed=failed)
