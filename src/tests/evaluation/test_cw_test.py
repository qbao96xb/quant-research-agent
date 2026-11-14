import numpy as np
import pandas as pd
import pytest

from src.evaluation.cw_test import cw_table, cw_test


def make(n=200, seed=0, signal=0.01):
    """Candidate A knows part of the return; benchmark B predicts zero."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    x = rng.normal(size=n)
    actual = signal * x + rng.normal(scale=0.01, size=n)
    fa = signal * x
    fb = np.zeros(n)
    return (pd.Series(fa - actual, index=idx), pd.Series(fb - actual, index=idx),
            pd.Series(fa, index=idx), pd.Series(fb, index=idx))


def test_identical_forecasts_no_evidence():
    eA, eB, fA, fB = make()
    r = cw_test(eB, eB, fB, fB, h=1)
    assert r["verdict"] == "no evidence"
    assert r["p"] == 0.5


def test_true_signal_is_detected():
    eA, eB, fA, fB = make()
    r = cw_test(eA, eB, fA, fB, h=1)
    assert r["p"] < 0.05
    assert r["verdict"] == "A better"
    assert r["cw"] > 0


def test_short_sample_insufficient():
    eA, eB, fA, fB = make(n=5)
    r = cw_test(eA, eB, fA, fB, h=1)
    assert r["verdict"] == "insufficient data"
    assert r["n"] == 5


def test_keys_and_default_lags():
    eA, eB, fA, fB = make()
    r = cw_test(eA, eB, fA, fB, h=5, step=1)
    assert set(r) == {"cw", "p", "n", "mean_adj", "verdict", "hac_lags"}
    assert r["hac_lags"] == 4
    assert r["n"] == 200


def results(n=60, seed=1):
    """Walk-forward style results: naive (zero), arima (knows signal), kronos_mini."""
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2021-01-04", periods=n, freq="7D")
    last = 100 * np.exp(rng.normal(scale=0.01, size=n).cumsum())
    signal = rng.normal(scale=0.01, size=n)
    actual = last * np.exp(signal + rng.normal(scale=0.01, size=n))
    base = dict(ticker="KO", h=5, origin=origins,
                target_date=origins + pd.Timedelta(days=7), last=last, actual=actual)
    return {"naive": pd.DataFrame(dict(base, pred=last)),
            "arima": pd.DataFrame(dict(base, pred=last * np.exp(signal))),
            "kronos_mini": pd.DataFrame(dict(base, pred=last * np.exp(0.5 * signal)))}


def test_cw_table_nested_models_only():
    tab = cw_table(results(), benchmark="naive", h=5, step=5)
    assert list(tab.index) == [("KO", "arima")]
    assert list(tab.columns) == ["h", "cw", "p", "n", "mean_adj", "verdict", "hac_lags"]
    row = tab.loc[("KO", "arima")]
    assert row["n"] == 60
    assert row["hac_lags"] == 0
    assert row["verdict"] == "A better"


def test_cw_table_missing_benchmark():
    r = results()
    del r["naive"]
    with pytest.raises(ValueError):
        cw_table(r, benchmark="naive", h=5, step=5)


def test_cw_table_inconsistent_targets():
    r = results()
    r["arima"] = r["arima"].assign(actual=r["arima"]["actual"] * 1.01)
    with pytest.raises(ValueError):
        cw_table(r, benchmark="naive", h=5, step=5)


def test_cw_table_skips_non_nested_only_run():
    r = results()
    del r["arima"]
    tab = cw_table(r, benchmark="naive", h=5, step=5)
    assert len(tab) == 0
