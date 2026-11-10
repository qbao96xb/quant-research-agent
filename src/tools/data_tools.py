# src/tools/data_tools.py
import numpy as np
import pandas as pd

from data.loader import fetch_ohlcv
from .cache import put, get_dataset, clean_json, data_fp
from .errors import ToolError
from .validation import check_ticker, check_dates
from .registry_core import tool
from .schemas import LoadDataArgs, DescribeArgs

TRADING_DAYS = 252
MIN_ROWS = 60           # minimum history for forecasting and volatility


def _date_str(x) -> str:
    return str(pd.Timestamp(x).date())

def _valid_close(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with a finite, strictly positive close."""
    close = pd.to_numeric(df["close"], errors="coerce")
    ok = np.isfinite(close) & (close > 0)
    return df[ok].reset_index(drop=True)

@tool(LoadDataArgs)
def load_data(args) -> dict:
    """Load daily adjusted OHLCV for one ticker (from cache or snapshot).

    Use first, before any forecast or backtest. Returns a dataset_id.
    Reuse the returned dataset_id. Do not call this twice for the same ticker.
    """
    ticker = check_ticker(args.ticker)
    s, e = check_dates(args.start, args.end)

    try:
        df = fetch_ohlcv(ticker, start=args.start, end=args.end)
    except ValueError as exc:                       # bad or empty data from the loader
        raise ToolError("DATA_FETCH_FAILED", str(exc),
                        "Check the ticker or try other dates.") from exc
    except Exception as exc:                        # network, parquet, yfinance errors
        raise ToolError("DATA_FETCH_FAILED", f"Could not load {ticker}: {exc}",
                        "Check the ticker and your connection.") from exc

    if df is None or len(df) == 0:
        raise ToolError("DATA_FETCH_FAILED", f"No data for {ticker}",
                        "Try another ticker or dates.")

    ts = pd.to_datetime(df["timestamps"])
    latest = _date_str(ts.max())  # capture before filtering
    mask = ts >= s
    if e is not None:
        mask &= ts < e  # end is exclusive, as in yfinance
    df = _valid_close(df[mask].reset_index(drop=True))

    if len(df) == 0:
        raise ToolError("DATA_FETCH_FAILED",
                        f"No data for {ticker} in the range {args.start} to {args.end or 'today'}. "
                        f"Latest available date is {latest}.",
                        "Use an earlier start date or a later end date.")

    first, last = _date_str(df["timestamps"].iloc[0]), _date_str(df["timestamps"].iloc[-1])
    aid = put("ds", df, dict(ticker=ticker, start=args.start, end=args.end,
                             rows=len(df), last=last, fp=data_fp(df)))
    return clean_json(dict(ok=True, dataset_id=aid, ticker=ticker, rows=len(df),
                           first=first, last=last,
                           last_close=float(df["close"].iloc[-1])))

@tool(DescribeArgs)
def describe_data(args) -> dict:
    """Summarize a loaded dataset: date range, last close, 20-day and 1-year volatility."""
    try:
        df = get_dataset(args.dataset_id)
    except (KeyError, FileNotFoundError) as exc:
        raise ToolError("NOT_FOUND", f"Unknown dataset_id {args.dataset_id}",
                        "Call load_data first and use the returned dataset_id.") from exc

    df = _valid_close(df)
    if len(df) < MIN_ROWS:
        raise ToolError("NOT_ENOUGH_DATA", f"Only {len(df)} valid rows",
                        "Reload with an earlier start date.")

    r = np.log(df["close"]).diff().dropna()
    ann = np.sqrt(TRADING_DAYS)
    return clean_json(dict(ok=True, rows=len(df),
                           first=_date_str(df["timestamps"].iloc[0]),
                           last=_date_str(df["timestamps"].iloc[-1]),
                           last_close=float(df["close"].iloc[-1]),
                           vol_20d_ann=float(r.tail(20).std() * ann),
                           vol_1y_ann=float(r.tail(TRADING_DAYS).std() * ann)))
