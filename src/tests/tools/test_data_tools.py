import json
import pytest
import pandas as pd
from src.tools.registry import call


def test_load_ok():
    r = call("load_data", ticker="KO", start="2015-01-01")
    assert r["ok"] and r["dataset_id"].startswith("ds_") and r["rows"] > 1000
    json.dumps(r)


@pytest.mark.parametrize("kw,code", [
    (dict(ticker="ko!!"), "BAD_TICKER"),
    (dict(ticker="TOOLONGX"), "BAD_TICKER"),
    (dict(ticker="KO", start="not-a-date"), "BAD_DATE"),
    (dict(ticker="KO", start="2020-01-01", end="2019-01-01"), "BAD_DATE"),
    (dict(ticker="KO", start="2020-01-01", end="2999-01-01"), "BAD_DATE"),
])
def test_load_bad_inputs(kw, code):
    r = call("load_data", **kw)
    assert not r["ok"] and r["error"]["code"] == code


def test_load_unknown_ticker():
    r = call("load_data", ticker="ZZZZ", start="2015-01-01")
    assert not r["ok"] and r["error"]["code"] == "DATA_FETCH_FAILED"


def test_short_history():
    start = (pd.Timestamp.today() - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    r = call("load_data", ticker="KO", start=start)
    assert r["ok"], r
    assert r["rows"] < 305
    f = call("forecast", dataset_id=r["dataset_id"], model="naive", h=5)
    assert not f["ok"] and f["error"]["code"] == "NOT_ENOUGH_DATA"

def test_describe():
    ds = call("load_data", ticker="KO", start="2015-01-01")["dataset_id"]
    r = call("describe_data", dataset_id=ds)
    assert r["ok"] and r["vol_1y_ann"] > 0
    assert call("describe_data", dataset_id="ds_nope")["error"]["code"] == "UNKNOWN_ID"
    assert call("describe_data", dataset_id="bt_nope")["error"]["code"] == "UNKNOWN_ID"
