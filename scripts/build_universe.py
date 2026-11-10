import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from data.loader import fetch_ohlcv

SEED = 1340
CUTOFF = "2024-12-31"
MIN_ROWS = 3750
MIN_PRICE = 5.0
HELD_OUT = {"MU", "KO"}
STRICT = False
SUBSTITUTES = {"PEP", "KDP", "AMAT", "LRCX", "WDC", "STX"}
N_PER_CELL = 2
N_CORE_MAX = 50
N_RALLY = 12
N_VAL = 8

# 1. output folder (must be defined before CACHE)
OUT = Path(__file__).resolve().parents[1] / "data" / "universe"
OUT.mkdir(parents=True, exist_ok=True)

# 2. universe with sectors (cached)
URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CACHE = OUT / "sp500.csv"
FEAT_CACHE = OUT / "features.csv"

if CACHE.exists():
    sp = pd.read_csv(CACHE)
else:
    r = requests.get(
        URL,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
        timeout=30,
    )
    r.raise_for_status()
    sp = pd.read_html(StringIO(r.text))[0]
    sp.to_csv(CACHE, index=False)
    print("saved", CACHE)

sp["Symbol"] = sp["Symbol"].str.replace(".", "-", regex=False)
sector = dict(zip(sp["Symbol"], sp["GICS Sector"]))

# 3. features per ticker, using data up to CUTOFF only (cached)
if FEAT_CACHE.exists():
    feat = pd.read_csv(FEAT_CACHE, index_col="ticker")
    print("loaded features cache:", len(feat))
else:
    cutoff_ts = pd.Timestamp(CUTOFF)
    rows = []
    for t in sector:
        try:
            d = fetch_ohlcv(t)
            d["timestamps"] = pd.to_datetime(d["timestamps"]).dt.tz_localize(None)
            d = d[d["timestamps"] <= cutoff_ts].dropna(subset=["close"])
            if len(d) < MIN_ROWS or d["close"].iloc[-1] < MIN_PRICE:
                continue
            c = d["close"]
            vol = c.pct_change(fill_method=None).tail(252).std() * np.sqrt(252)
            rally = (c.pct_change(20, fill_method=None) > 0.20).mean()
            rows.append(dict(ticker=t, sector=sector[t], vol=vol, rally=rally, n=len(d)))
        except Exception as e:
            print("skip", t, e)
    feat = pd.DataFrame(rows).set_index("ticker")
    feat.to_csv(FEAT_CACHE)
    print("saved", FEAT_CACHE)

# 4. exclusions (applied after cache, so toggling STRICT needs no refetch)
drop = set(HELD_OUT) | (SUBSTITUTES if STRICT else set())
feat = feat.drop(index=[t for t in drop if t in feat.index])
print("eligible:", len(feat))

# 5. validation tickers (random, never trained on)
val = feat.sample(n=N_VAL, random_state=SEED).index.tolist()
pool = feat.drop(index=val).copy()

# 6. stratified core (List B)
pool["vol_bucket"] = pd.qcut(pool["vol"], 3, labels=["low", "mid", "high"], duplicates="drop")

picked = []
for _, g in pool.groupby(["sector", "vol_bucket"], observed=True):
    picked += g.sample(n=min(N_PER_CELL, len(g)), random_state=SEED).index.tolist()

if len(picked) > N_CORE_MAX:
    rng = np.random.default_rng(SEED)
    picked = rng.choice(picked, size=N_CORE_MAX, replace=False).tolist()
core = picked

# 7. rally add-on (List C)
rest = pool.drop(index=core)
rally = rest.sort_values("rally", ascending=False).head(N_RALLY).index.tolist()

list_c = sorted(set(core) | set(rally))

# 8. diagnostics
print("List C size:", len(list_c))
print(feat.loc[list_c].groupby("sector").size())
print("median vol  all:", round(feat["vol"].median(), 2),
      " list C:", round(feat.loc[list_c, "vol"].median(), 2))
print("mean rally share  all:", round(feat["rally"].mean(), 3),
      " list C:", round(feat.loc[list_c, "rally"].mean(), 3))

with open(OUT / "list_c.json", "w") as f:
    json.dump(
        dict(seed=SEED, cutoff=CUTOFF, held_out=sorted(HELD_OUT), strict=STRICT,
             validation=sorted(val), core=sorted(core), rally=sorted(rally),
             list_c=list_c),
        f, indent=2,
    )
print("saved", OUT / "list_c.json")
