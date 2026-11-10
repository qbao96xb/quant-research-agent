# src/tests/tools/test_forecast_path.py
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.tools import forecast_tools as ft
from src.tools.registry import call as registry_call, schemas
from src.tools.schemas import ForecastPathArgs

LAST_CLOSE = 71.0


def call(**kw):
    args = dict(dataset_id="ds_x", model="naive", h=5)
    args.update(kw)
    return registry_call("forecast_path", **args)


@pytest.fixture
def hist():
    n = 600
    return pd.DataFrame({
        "timestamps": pd.bdate_range("2023-01-01", periods=n),
        "close": np.full(n, LAST_CLOSE),
    })

class Dummy:
    """Fake model returning a wavy path, so daily and cumulative returns differ."""
    def __init__(self, length=None):
        self.length = length

    def forecast(self, hist, h):
        k = self.length or h
        mean = pd.Series(LAST_CLOSE + np.sin(np.arange(1, k + 1)))
        return type("FC", (), dict(mean=mean, lower=mean - 1.0, upper=mean + 1.0))()


@pytest.fixture
def patched(monkeypatch, hist):
    monkeypatch.setattr(ft, "get_dataset", lambda _id: hist)
    monkeypatch.setattr(ft, "make_model", lambda name, order=None: Dummy())
    monkeypatch.setattr(ft, "add_intervals", lambda fc, *a, **k: fc)


# ---------- schema ----------
def test_schema_defaults():
    a = ForecastPathArgs(dataset_id="ds_x", model="naive")
    assert a.h == 10 and a.origin is None


@pytest.mark.parametrize("h", [0, 21])
def test_schema_rejects_bad_h(h):
    with pytest.raises(ValidationError):
        ForecastPathArgs(dataset_id="ds_x", model="naive", h=h)


def test_tool_is_registered():
    assert "forecast_path" in {s["name"] for s in schemas()}


# ---------- behavior ----------
def test_path_length_and_fields(patched):
    out = call(h=5)
    assert out["ok"]
    assert out["h"] == 5 and len(out["path"]) == 5
    assert [d["day"] for d in out["path"]] == [1, 2, 3, 4, 5]

    OLD_KEYS = {"day", "price", "cum_log_return", "daily_log_return", "lo", "hi"}
    PCT_KEYS = {"daily_pct", "cum_pct", "cum_lo_pct", "cum_hi_pct"}
    ID_KEYS = {"ticker", "model"}
    for d in out["path"]:
        assert set(d) == OLD_KEYS | PCT_KEYS | ID_KEYS
        assert d["model"] == out["model"]
        assert d["ticker"] == out["ticker"]

    assert out["last_close"] == pytest.approx(LAST_CLOSE)

def test_daily_returns_sum_to_cumulative(patched):
    path = call(h=10)["path"]
    total = sum(d["daily_log_return"] for d in path)
    # 6-dp rounding per day: allow up to h * 0.5e-6
    assert total == pytest.approx(path[-1]["cum_log_return"], abs=len(path) * 1e-6)

def test_first_day_daily_equals_cumulative(patched):
    d1 = call(h=3)["path"][0]
    assert d1["daily_log_return"] == pytest.approx(d1["cum_log_return"])

def test_cum_return_matches_price(patched):
    for d in call(h=4)["path"]:
        # price and return are each rounded, so compare at ~1e-5
        assert d["cum_log_return"] == pytest.approx(np.log(d["price"] / LAST_CLOSE), abs=1e-5)

def test_interval_brackets_mean(patched):
    for d in call(h=6)["path"]:
        assert d["lo"] < d["cum_log_return"] < d["hi"]


def test_length_mismatch_fails_cleanly(monkeypatch, patched):
    monkeypatch.setattr(ft, "make_model", lambda name: Dummy(length=3))
    out = call(h=5)
    assert not out.get("ok", False)
    assert out["error"]["code"] != "UNKNOWN_TOOL"

def test_bad_args_return_error_not_crash(patched):
    out = call(h=99)
    assert not out.get("ok", False)
    assert out["error"]["code"] == "BAD_ARGUMENT"


def test_order_is_passed_to_make_model(monkeypatch, hist):
    seen = {}
    monkeypatch.setattr(ft, "get_dataset", lambda _id: hist)
    monkeypatch.setattr(ft, "make_model", lambda name, order=None: seen.update(name=name, order=order) or Dummy())
    monkeypatch.setattr(ft, "add_intervals", lambda fc, *a, **k: fc)
    out = call(model="arima", order=[2, 0, 1])
    assert out["ok"] and seen == {"name": "arima", "order": (2, 0, 1)}
    assert out["order"] == [2, 0, 1]
