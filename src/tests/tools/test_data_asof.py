import os

import pandas as pd

import data.loader as loader
from data.asof import DATA_ASOF, cap_asof, set_asof


def frame(days):
    ts = pd.bdate_range("2026-11-02", periods=days)
    return pd.DataFrame({"timestamps": ts, "open": 1.0, "high": 1.0, "low": 1.0,
                         "close": 1.0, "volume": 1.0, "amount": 1.0})


def test_asof_cuts_later_rows(monkeypatch):
    monkeypatch.setenv("QUANT_DATA_ASOF", "2026-11-06")
    out = cap_asof(frame(10))
    assert str(out["timestamps"].max())[:10] == "2026-11-06"
    assert len(out) == 5


def test_no_asof_keeps_all_rows(monkeypatch):
    monkeypatch.delenv("QUANT_DATA_ASOF", raising=False)
    assert len(cap_asof(frame(10))) == 10


def test_cached_path_applies_asof(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_DATA_ASOF", "2026-11-06")
    monkeypatch.setattr(loader, "CACHE", tmp_path)
    monkeypatch.setattr(loader, "SNAPSHOTS", tmp_path / "snapshots")
    monkeypatch.setattr(loader, "_fresh", lambda f, df: True)
    frame(10).to_parquet(tmp_path / "ZZ.parquet", index=False)
    out = loader.fetch_ohlcv("ZZ", start="2026-11-02")
    assert len(out) == 5


def test_set_asof_sets_and_clears(monkeypatch):
    monkeypatch.delenv("QUANT_DATA_ASOF", raising=False)
    assert set_asof("2026-11-06") == "2026-11-06"
    assert os.environ["QUANT_DATA_ASOF"] == "2026-11-06"
    assert set_asof("none") is None
    assert "QUANT_DATA_ASOF" not in os.environ


def test_eval_default_is_pinned():
    assert DATA_ASOF == "2026-11-06"
