import numpy as np
import pandas as pd
import pytest

import src.tools.tune_arima_tools as tt
from src.tools import cache
from src.tools.registry import call

FAST = dict(lookback=750, refit_every=10)



def _df(n=1500, seed=0):
    r = np.random.default_rng(seed).normal(0, 0.01, n)
    return pd.DataFrame({"timestamps": pd.bdate_range("2015-01-01", periods=n),
                         "close": 100 * np.exp(np.cumsum(r))})



def _ds(n=1500, seed=0):
    return cache.put("ds", _df(n, seed), {"test": "tune_tool", "n": n, "seed": seed})



@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "STORE", tmp_path)
    monkeypatch.setattr(cache, "_MEM", {})
    real = tt._tune
    monkeypatch.setattr(tt, "_tune", lambda df, **kw: real(df, cfg=FAST, **kw))



def test_not_enough_data_is_tool_error():
    r = call("tune_arima", dataset_id=_ds(500))
    assert r["ok"] is False
    assert r["error"]["code"] == "NOT_ENOUGH_DATA"
    assert r["error"]["hint"]



def test_output_keys():
    r = call("tune_arima", dataset_id=_ds(), max_p=1, max_q=1)
    assert r["ok"] is True, r
    for k in ("best_order", "table", "test_rmse", "held_out_check",
              "check_reason", "cfg", "tune_id"):
        assert k in r, k
    assert r["held_out_check"] in ("pass", "fail")
    assert r["cfg"] == FAST
    assert r["tune_id"].startswith("tune_")



def test_tune_id_stable_and_changes():
    ds = _ds()
    a = call("tune_arima", dataset_id=ds, max_p=1, max_q=1)
    b = call("tune_arima", dataset_id=ds, max_p=1, max_q=1)
    c = call("tune_arima", dataset_id=ds, max_p=1, max_q=1, n_val=50)
    d = call("tune_arima", dataset_id=_ds(seed=1), max_p=1, max_q=1)
    assert a["ok"] and b["ok"] and c["ok"] and d["ok"]
    assert a["tune_id"] == b["tune_id"]
    assert len({a["tune_id"], c["tune_id"], d["tune_id"]}) == 3


def test_retune_attempt_and_note():
    ds = _ds()
    a = call("tune_arima", dataset_id=ds, max_p=1, max_q=1)
    a2 = call("tune_arima", dataset_id=ds, max_p=1, max_q=1)
    b = call("tune_arima", dataset_id=ds, max_p=1, max_q=1, n_val=50)
    assert a["ok"] and a2["ok"] and b["ok"]
    assert a["attempt"] == 1 and a2["attempt"] == 1
    assert tt.RETUNE_NOTE not in a["note"]
    assert b["attempt"] == 2
    assert tt.RETUNE_NOTE in b["note"]
    ids = [x["id"] for x in call("list_artifacts")["artifacts"]]
    assert not any(i.startswith("tune_index") for i in ids)