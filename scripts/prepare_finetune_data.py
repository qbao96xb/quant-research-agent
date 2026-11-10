from pathlib import Path
import pandas as pd, yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "finetune"; OUT.mkdir(parents=True, exist_ok=True)
TICKERS = ["KO", "PEP", "PG", "WMT", "JNJ", "MU", "AAPL", "MSFT"]  # extend to ~100-500

for t in TICKERS:
    df = yf.download(t, start="2000-01-01", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df = df.reset_index().rename(columns={"Date": "timestamps", "date": "timestamps"})
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df["amount"] = df["close"] * df["volume"]
    df[["timestamps", "open", "high", "low", "close", "volume", "amount"]].to_csv(OUT / f"{t}.csv", index=False)
    print(t, len(df))
