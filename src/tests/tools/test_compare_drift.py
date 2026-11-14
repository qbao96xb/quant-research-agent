import json

import pytest

from src.tools.registry import call


@pytest.fixture(scope="module")
def bt():
    ds = call("load_data", ticker="KO", start="2015-01-01")["dataset_id"]
    r = call("run_backtest", dataset_id=ds, models=["naive", "drift", "arima"], h=5, start=-500, step=5)
    assert r["ok"], r
    return r["backtest_id"]



def cells(table, model):
    row = [l for l in table.splitlines() if l.startswith(f"| {model} |")][0]
    return [c.strip() for c in row.strip().strip("|").split("|")]



def test_header_has_drift_column_last(bt):
    r = call("compare_models", backtest_id=bt)
    assert r["ok"], r
    json.dumps(r)
    assert r["table"].splitlines()[0].endswith("| DM p-value | CW p-value | DM p vs drift |")



def test_drift_cells(bt):
    t = call("compare_models", backtest_id=bt)["table"]
    assert cells(t, "naive")[-1] == "-"
    assert cells(t, "drift")[-1] == "-"
    p = float(cells(t, "arima")[-1])
    assert 0.0 <= p <= 1.0
    assert 0.0 <= float(cells(t, "drift")[-2]) <= 1.0



def test_note_mentions_drift_reference(bt):
    r = call("compare_models", backtest_id=bt)
    assert "DM p vs drift uses drift as a reference benchmark." in r["note"]



def test_drift_benchmark_has_no_drift_column(bt):
    r = call("compare_models", backtest_id=bt, benchmark="drift")
    assert r["ok"], r
    assert "DM p vs drift" not in r["table"]
    assert "reference benchmark" not in r["note"]
