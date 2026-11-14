"""Price data loader with a local cache, dated snapshots and a manifest.

Layout
------
data/raw/
    {ticker}.parquet                          # working cache (latest full download)
    snapshots/{ticker}_{YYYY-MM-DD}.parquet   # immutable dated copies
    manifest.csv                              # one row per download

Rules
-----
- Rows with NaN or non-positive prices are dropped before anything is saved.
- Today's bar is dropped, because it can be partial while the market is open.
- The working cache is reused only if it is fresh, reaches back to `start`,
  and the request has no `end`.
- A request with `end` always downloads and never overwrites the working cache.

Example
-------
>>> df = fetch_ohlcv("KO")                           # cached
>>> df = fetch_ohlcv("KO", refresh=True)             # re-download + new snapshot
>>> df = load_snapshot("KO", "2025-06-30")           # exact reproduction
"""
import hashlib
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .asof import cap_asof

CACHE = Path(__file__).resolve().parents[2] / "data" / "raw"
SNAPSHOTS = CACHE / "snapshots"
MANIFEST = CACHE / "manifest.csv"
MAX_AGE_DAYS = 1       # re-download if the cache file is older than this
MAX_BAR_AGE = 4        # re-download if the last bar is older (covers weekends)
REQUIRED = ["open", "high", "low", "close", "volume"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bad rows, duplicates and today's (possibly partial) bar."""
    df = df.copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.dropna(subset=["close", "open", "high", "low"])
    df = df[np.isfinite(df["close"]) & (df["close"] > 0)]
    df = df.drop_duplicates("timestamps").sort_values("timestamps")
    df = df[df["timestamps"] < pd.Timestamp(date.today())]
    return df.reset_index(drop=True)


def _fresh(f: Path, df: pd.DataFrame) -> bool:
    """True if the cache file is recent and its last bar is recent."""
    bar_age = (date.today() - pd.Timestamp(df["timestamps"].iloc[-1]).date()).days
    file_age = time.time() - f.stat().st_mtime
    return bar_age <= MAX_BAR_AGE and file_age < MAX_AGE_DAYS * 86400


def _write_manifest(ticker, df, start, end, snap_path):
    """Append one row describing a download."""
    row = dict(
        ticker=ticker,
        downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        request_start=start,
        request_end=end,
        first_date=str(pd.Timestamp(df["timestamps"].iloc[0]).date()),
        last_date=str(pd.Timestamp(df["timestamps"].iloc[-1]).date()),
        n_rows=len(df),
        sha256=hashlib.sha256(
            pd.util.hash_pandas_object(df, index=False).values.tobytes()
        ).hexdigest()[:16],
        file=snap_path.name,
        source="yfinance",
        auto_adjust=True,
    )
    pd.DataFrame([row]).to_csv(
        MANIFEST, mode="a", header=not MANIFEST.exists(), index=False
    )


def fetch_ohlcv(ticker: str, start="2000-01-01", end=None, refresh=False) -> pd.DataFrame:
    """Return daily OHLCV for `ticker`, downloading only if needed.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol, e.g. "KO".
    start : str, default "2000-01-01"
        First date requested.
    end : str, optional
        Last date requested (exclusive in yfinance). None means up to yesterday.
        If given, the cache is bypassed and the working cache is not overwritten.
    refresh : bool, default False
        True forces a new download, a new snapshot and a manifest row.

    Returns
    -------
    pandas.DataFrame
        Columns: timestamps, open, high, low, close, volume, amount.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{ticker}.parquet"

    # 1. Try the cache
    if f.exists() and not refresh and end is None:
        cached = pd.read_parquet(f)
        if len(cached):
            covers = (pd.Timestamp(cached["timestamps"].iloc[0])
                      <= pd.Timestamp(start) + pd.Timedelta(days=7))
            if covers and _fresh(f, cached):
                return cap_asof(cached)

    # 2. Download
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker}")
    raw.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                   for c in raw.columns]
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns for {ticker}: {missing}")
    df = raw.reset_index().rename(columns={"Date": "timestamps", "date": "timestamps"})

    # 3. Clean and validate
    df = _clean(df)
    if df.empty:
        raise ValueError(f"No valid prices for {ticker}")
    df["amount"] = df["close"] * df["volume"]

    # 4. Save: dated snapshot always, working cache only for open-ended requests
    stamp = date.today().strftime("%Y-%m-%d")
    snap = SNAPSHOTS / f"{ticker}_{stamp}.parquet"
    df.to_parquet(snap, index=False)
    if end is None:
        df.to_parquet(f, index=False)
    _write_manifest(ticker, df, start, end, snap)
    return cap_asof(df)

def load_snapshot(ticker: str, snapshot_date: str) -> pd.DataFrame:
    """Load an exact dated snapshot, for reproducing a past run.

    Parameters
    ----------
    ticker : str
        Symbol.
    snapshot_date : str
        Snapshot date, "YYYY-MM-DD".
    """
    return pd.read_parquet(SNAPSHOTS / f"{ticker}_{snapshot_date}.parquet")
