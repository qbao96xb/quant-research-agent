import sys
from pathlib import Path

import pandas as pd
import torch
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "Kronos"))
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

TICKER, LOOKBACK, PRED_LEN = "MU", 500, 30
OUT = ROOT / "results" / "baseline"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Data
raw = yf.download(TICKER, period="3y", auto_adjust=True, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
raw.columns = [str(c).lower() for c in raw.columns]
raw = raw.reset_index().rename(columns={"Date": "timestamps", "date": "timestamps"})

cols = ["open", "high", "low", "close", "volume"]
raw = raw.dropna(subset=cols).reset_index(drop=True)
raw["amount"] = raw["close"] * raw["volume"]

hist = raw.iloc[-LOOKBACK:].reset_index(drop=True)
x_df = hist[["open", "high", "low", "close", "volume", "amount"]]

# debug (after x_df exists)
print(raw.columns.tolist())
print(x_df.isna().sum())

x_timestamp = pd.to_datetime(hist["timestamps"])
y_timestamp = pd.Series(
    pd.bdate_range(x_timestamp.iloc[-1] + pd.offsets.BDay(1), periods=PRED_LEN)
)


# 2. Model
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

# 3. Forecast
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=PRED_LEN,
    T=1.0,
    top_p=0.9,
    sample_count=10,
)

# 4. Save
pred_df.index = y_timestamp
hist.to_csv(OUT / "kronos_input.csv", index=False)
pred_df.to_csv(OUT / "kronos_forecast.csv")
print(device)
print(pred_df.head())
