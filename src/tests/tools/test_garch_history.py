import json

import pandas as pd
import pytest

from src.tools.cache import get_dataset
from src.tools.registry import call


@pytest.fixture(scope="module")
def ko():
    ds = call("load_data", ticker="KO", start="2015-01-01")["dataset_id"]
    out = call("garch_volatility", dataset_id=ds, h=5)
    assert out["ok"], out
    json.dumps(out)
    return ds, out


def test_history_has_60_positive_days(ko):
    _, out = ko
    hist = out["history"]
    assert len(hist) == 60
    assert all(set(r) == {"date", "cond_vol"} for r in hist)
    assert all(0 < r["cond_vol"] < 0.2 for r in hist)


def test_history_dates_end_at_origin(ko):
    ds, out = ko
    dates = pd.to_datetime([r["date"] for r in out["history"]])
    assert dates.is_monotonic_increasing and dates.is_unique
    last = pd.Timestamp(get_dataset(ds)["timestamps"].iloc[-1]).normalize()
    assert dates[-1] == last


def test_existing_keys_unchanged(ko):
    _, out = ko
    for k in ("ok", "h", "model", "ticker", "level", "sigma", "lo", "hi", "path"):
        assert k in out
    assert len(out["path"]) == 5
