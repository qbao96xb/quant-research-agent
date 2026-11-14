"""Which tickers fail tune_arima's held-out check on the pinned eval data? (eval case T02)

Makes the agent's T02 tool calls: load_data(end=END), then tune_arima with defaults.
Usage: python scripts/probe_held_out.py [--end DATE ...] TICKER [TICKER ...]
Example: python scripts/probe_held_out.py --end 2020-06-30 2022-06-30 -- XOM CVX
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from src.data.asof import DATA_ASOF, set_asof  # noqa: E402
from src.tools.registry import call            # noqa: E402

END = "2026-11-06"   # as in the T02 prompt ("through 2026-11-06"); load_data treats end as exclusive



def line(ticker, d, r):
    """One result line from the load_data result d and the tune_arima result r."""
    if not d.get("ok"):
        return f"{ticker}: load_data {d['error']['code']}"
    if not r.get("ok"):
        return f"{ticker}: tune_arima {r['error']['code']}"
    nv, df = r["dm_best_vs_naive"], r["dm_best_vs_default"]
    p_def = "-" if df is None else f"{df['p']:.3f}"
    return (f"{ticker}: {r['held_out_check']} best={tuple(r['best_order'])} "
            f"p_vs_naive={nv['p']:.3f} p_vs_101={p_def} rows={d['rows']} last={d['last']}")



def parse(argv):
    ap = argparse.ArgumentParser(description="Probe tune_arima's held-out check")
    ap.add_argument("--end", nargs="+", default=[END], help="end dates (exclusive), YYYY-MM-DD")
    ap.add_argument("tickers", nargs="+")
    return ap.parse_args(argv)



def probe(ticker, end):
    d = call("load_data", ticker=ticker, start="2015-01-01", end=end)
    r = call("tune_arima", dataset_id=d["dataset_id"]) if d.get("ok") else {}
    return line(f"{ticker} {end}", d, r)



def main(argv):
    a = parse(argv)
    set_asof(DATA_ASOF)
    print(f"data_asof {DATA_ASOF}, ends {' '.join(a.end)}", flush=True)
    for end in a.end:
        for t in a.tickers:
            try:
                print(probe(t, end), flush=True)
            except Exception as e:
                print(f"{t} {end}: error {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
