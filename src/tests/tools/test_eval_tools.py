import json
import pytest
from src.tools.registry import call


@pytest.fixture(scope="module")
def bt():
    ds = call("load_data", ticker="KO", start="2015-01-01")["dataset_id"]
    r = call("run_backtest", dataset_id=ds, models=["naive", "arima"], h=5, start=-500, step=5)
    assert r["ok"], r
    json.dumps(r)
    return r["backtest_id"]


def test_metrics(bt):
    r = call("get_metrics", backtest_id=bt, model="arima")
    assert r["ok"], r
    json.dumps(r)


def test_metrics_missing_model(bt):
    assert not call("get_metrics", backtest_id=bt, model="drift")["ok"]


def test_dm_same_model(bt):
    assert not call("dm_test", backtest_id=bt, model_a="naive", model_b="naive")["ok"]


def test_wrong_id_kind(bt):
    r = call("get_metrics", backtest_id="ds_abc", model="arima")
    assert r["error"]["code"] == "UNKNOWN_ID"


def test_garch():
    ds = call("load_data", ticker="KO", start="2015-01-01")["dataset_id"]
    assert call("garch_volatility", dataset_id=ds, h=5)["ok"]
    assert call("garch_volatility", dataset_id=ds, h=0)["error"]["code"] == "BAD_ARGUMENT"
    assert call("garch_volatility", dataset_id=ds, h=99)["error"]["code"] == "BAD_ARGUMENT"

def test_garch_path():
    ds = call("load_data", ticker="KO", start="2015-01-01")["dataset_id"]
    out = call("garch_volatility", dataset_id=ds, h=5)
    assert out["ok"], out
    json.dumps(out)
    assert len(out["path"]) == 5
    assert [d["day"] for d in out["path"]] == [1, 2, 3, 4, 5]
    assert out["path"][-1]["sigma"] == pytest.approx(out["sigma"])
    for d in out["path"]:
        assert set(d) == {"ticker", "day", "daily_sigma", "daily_sigma_pct",
                          "sigma", "cum_sigma_pct", "lo", "hi"}
        assert d["ticker"] == out["ticker"] == "KO"
        assert d["hi"] > 0 and d["lo"] == pytest.approx(-d["hi"])
    cum_var = sum(d["daily_sigma"] ** 2 for d in out["path"])
    assert cum_var == pytest.approx(out["sigma"] ** 2, rel=1e-4)


def test_garch_level():
    ds = call("load_data", ticker="KO", start="2015-01-01")["dataset_id"]
    a = call("garch_volatility", dataset_id=ds, h=5, level=0.90)
    b = call("garch_volatility", dataset_id=ds, h=5, level=0.95)
    assert a["ok"] and b["ok"]
    assert a["level"] == "90%" and b["level"] == "95%"
    assert b["hi"] > a["hi"]
    assert a["sigma"] == pytest.approx(b["sigma"])
