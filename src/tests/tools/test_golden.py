# tests/tools/test_golden.py
"""Golden test: ARIMA vs naive DM statistic must match the frozen phase 2 result.

Runs on a frozen KO price file kept with the tests (src/tests/fixtures), never on live data.
Live downloads are not reproducible: Yahoo's adjusted closes jitter by ~1e-6 between downloads,
which flips one ARIMA(1,0,1) forecast (2025-12-29) and moves DM by ~0.02. The fixture is only
read by this test; the loader and the agent never see it.

Fixture: the last 1600 daily rows of KO up to 2026-10-21 (backtest needs 500 origins + 1000
lookback). Expected values come from results/phase2/20261021_195633/dm.csv (KO, arima, h=5).
To regenerate the fixture, re-run phase 2 on new data and update EXPECTED too.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.tools import data_tools
from src.tools.registry import call

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "ko_20261021.parquet"
N_ROWS = 1600
START, END = "2020-06-10", "2026-10-22"       # END exclusive, last row 2026-10-21
EXPECTED = dict(dm=-0.9726396, n=99, verdict="no difference")


@pytest.fixture(scope="module")
def ds_ko():
    frozen = pd.read_parquet(FIXTURE)
    mp = pytest.MonkeyPatch()
    mp.setattr(data_tools, "fetch_ohlcv", lambda *a, **k: frozen.copy())
    try:
        r = call("load_data", ticker="KO", start=START, end=END)
        assert r["ok"], r
        assert r["rows"] == N_ROWS, f"fixture changed: {r['rows']} rows"
        yield r["dataset_id"]
    finally:
        mp.undo()


def test_dm_matches_phase2(ds_ko):
    bt = call("run_backtest", dataset_id=ds_ko, models=["naive", "arima"],
              h=5, start=-500, step=5)
    assert bt["ok"], bt
    r = call("dm_test", backtest_id=bt["backtest_id"], model_a="arima")
    assert r["ok"], r

    assert r["n"] == EXPECTED["n"]
    assert abs(r["dm"] - EXPECTED["dm"]) < 1e-3
    assert r["verdict"] == EXPECTED["verdict"]
