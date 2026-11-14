"""Optional cap on price data dates, so evals do not drift (QUANT_DATA_ASOF=YYYY-MM-DD)."""
import os

import pandas as pd

DATA_ASOF = "2026-11-06"   # eval default (owner, 2026-11-13); "none" means the latest data


def set_asof(value):
    """Set QUANT_DATA_ASOF for this process and its workers; "none" clears it. Returns the date or None."""
    if not value or value.lower() == "none":
        os.environ.pop("QUANT_DATA_ASOF", None)
        return None
    os.environ["QUANT_DATA_ASOF"] = value
    return value


def cap_asof(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows after QUANT_DATA_ASOF when it is set; return df unchanged otherwise."""
    asof = os.environ.get("QUANT_DATA_ASOF")
    if not asof:
        return df
    keep = pd.to_datetime(df["timestamps"]) <= pd.Timestamp(asof)
    return df[keep].reset_index(drop=True)
