# tests/tools/test_forecast_tools.py
import json, pytest
from src.tools.registry import call

@pytest.fixture(scope="module")
def ds():
    r = call("load_data", ticker="KO", start="2015-01-01")
    assert r["ok"], r
    return r["dataset_id"]

def test_no_lookahead(ds):
    a = call("forecast", dataset_id=ds, model="drift", h=5, origin="2024-06-03")
    assert a["ok"], a
    assert str(a["origin"])[:10] <= "2024-06-03"

def test_forecast_happy(ds):
    r = call("forecast", dataset_id=ds, model="drift", h=5)
    assert r["ok"] and r["lo"] < r["mean"] < r["hi"]
    json.dumps(r)

@pytest.mark.parametrize("kw,code", [
    (dict(h=0), "BAD_ARGUMENT"),
    (dict(h=99), "BAD_ARGUMENT"),
    (dict(model="lstm"), "BAD_ARGUMENT"),
    (dict(origin="2999-01-01"), "BAD_DATE"),
])
def test_forecast_bad_inputs(ds, kw, code):
    r = call("forecast", **{**dict(dataset_id=ds, model="naive", h=5), **kw})
    assert not r["ok"] and r["error"]["code"] == code

def test_unknown_id():
    r = call("forecast", dataset_id="ds_nope", model="naive", h=5)
    assert r["error"]["code"] == "UNKNOWN_ID"

def test_no_lookahead(ds):
    a = call("forecast", dataset_id=ds, model="drift", h=5, origin="2024-06-03")
    assert a["origin"] <= "2024-06-03"
