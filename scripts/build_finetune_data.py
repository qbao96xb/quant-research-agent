from pathlib import Path
from data.loader import fetch_ohlcv
import json

U = json.load(open(Path(__file__).resolve().parents[1] / "data" / "universe" / "list_c.json"))
TICKERS = U["list_c"]
HELD_OUT = set(U["held_out"]) | set(U["validation"])

START, T2 = "2005-01-01", "2024-12-31"
COLS = ["timestamps","open","high","low","close","volume","amount"]
OUT = Path(__file__).resolve().parents[1] / "data" / "finetune"
OUT.mkdir(parents=True, exist_ok=True)

# clear all old csv files, including any combined file
for p in OUT.glob("*.csv"):
    p.unlink()

total = 0
for t in TICKERS:
    if t in HELD_OUT:
        continue
    df = fetch_ohlcv(t)
    df["timestamps"] = df["timestamps"].astype(str)
    df = df[(df["timestamps"] >= START) & (df["timestamps"] <= T2)][COLS].dropna()
    if len(df) < 500:
        print(f"skip {t}: only {len(df)} rows")
        continue
    df.to_csv(OUT / f"{t}.csv", index=False)
    total += len(df)
    print(t, len(df))

print("Total rows:", total)
print("Files:", sorted(p.name for p in OUT.glob("*.csv")))
