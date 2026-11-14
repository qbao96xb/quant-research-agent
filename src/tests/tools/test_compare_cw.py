import json

import pytest

from src.tools.registry import call


@pytest.fixture(scope="module")
def bt():
    ds = call("load_data", ticker="KO", start="2015-01-01")["dataset_id"]
    r = call("run_backtest", dataset_id=ds, models=["naive", "arima"], h=5, start=-500, step=5)
    assert r["ok"], r
    return r["backtest_id"]


def cells(table, model):
    row = [l for l in table.splitlines() if l.startswith(f"| {model} |")][0]
    return [c.strip() for c in row.strip().strip("|").split("|")]


def test_header_has_cw_last(bt):
    r = call("compare_models", backtest_id=bt)
    assert r["ok"], r
    json.dumps(r)
    assert r["table"].splitlines()[0].endswith("| DM p-value | CW p-value |")


def test_cw_cells_naive_dash_arima_number(bt):
    t = call("compare_models", backtest_id=bt)["table"]
    assert cells(t, "naive")[-1] == "-"
    p = float(cells(t, "arima")[-1])
    assert 0.0 <= p <= 1.0


def test_note_mentions_cw_vs_naive(bt):
    r = call("compare_models", backtest_id=bt)
    assert "CW p-value (one-sided, nested models only) is vs naive." in r["note"]


def test_non_naive_benchmark_has_no_cw(bt):
    r = call("compare_models", backtest_id=bt, benchmark="arima")
    assert r["ok"], r
    assert cells(r["table"], "naive")[-1] == "-"
    assert cells(r["table"], "arima")[-1] == "-"
    assert "CW" not in r["note"]
