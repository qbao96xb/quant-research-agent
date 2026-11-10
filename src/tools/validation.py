# src/tools/validation.py
import re
import pandas as pd
from .errors import ToolError

TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z])?$")
MAX_H = 20
MIN_ROWS = 300

def check_ticker(t: str) -> str:
    t = t.strip().upper()
    if not TICKER_RE.match(t):
        raise ToolError("BAD_TICKER", f"{t!r} is not a valid ticker format",
                        "Use 1 to 5 letters, for example KO or MU.")
    return t

def check_dates(start, end=None):
    today = pd.Timestamp.today().normalize()
    s = pd.to_datetime(start, errors="coerce")
    e = pd.to_datetime(end, errors="coerce") if end else None
    if pd.isna(s) or (end and pd.isna(e)):
        raise ToolError("BAD_DATE", f"Could not parse dates: {start!r}, {end!r}", "Use YYYY-MM-DD.")
    if s > today:
        raise ToolError("BAD_DATE", f"Start date {start} is in the future. No data is available yet.",
                        "Use a start date in the past.")
    if e is not None and e <= s:
        raise ToolError("BAD_DATE", f"end {end} is not after start {start}", "Swap or fix the dates.")
    if e is not None and e > today:
        raise ToolError("BAD_DATE", f"end {end} is in the future", "Use today or earlier.")
    return s, e

def check_horizon(h: int, n_rows: int):
    if not 1 <= h <= MAX_H:
        raise ToolError("BAD_HORIZON", f"h={h} outside 1 to {MAX_H}", f"Choose h between 1 and {MAX_H}.")
    if n_rows < MIN_ROWS + h:
        raise ToolError("NOT_ENOUGH_DATA", f"Only {n_rows} rows for h={h}",
                        f"Need at least {MIN_ROWS + h}. Use an earlier start date.")

def check_origin(hist, origin, h):
    if origin is None:
        return hist
    o = pd.to_datetime(origin, errors="coerce")
    if pd.isna(o) or o < hist["timestamps"].iloc[MIN_ROWS]:
        raise ToolError("BAD_DATE", f"Origin {origin!r} too early or invalid", "Pick a later origin date.")
    return hist[hist["timestamps"] <= o]    # never leak data after the origin
