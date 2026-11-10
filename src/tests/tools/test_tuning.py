import numpy as np, pandas as pd
from evaluation.tuning import make_splits, tune_arima

def _df(n=1500, seed=0):
    r = np.random.default_rng(seed).normal(0, 0.01, n)
    return pd.DataFrame({"timestamps": pd.bdate_range("2015-01-01", periods=n),
                         "close": 100 * np.exp(np.cumsum(r))})

def test_blocks_do_not_overlap():
    df = _df(); h = 5
    (vdf, vs), (tdf, ts) = make_splits(df, h, 60, 40)
    assert vs + 59 * h + h <= ts                   # last val target <= first test origin
    assert len(vdf) == ts + 1

def test_deterministic_and_complete():
    a = tune_arima(_df(), max_p=1, max_q=1)
    b = tune_arima(_df(), max_p=1, max_q=1)
    assert a["best_order"] == b["best_order"] and a["orders_tried"] == 4
    assert a["note"] and a["n_test"] == 40

def test_short_history_raises():
    import pytest
    with pytest.raises(ValueError, match="NOT_ENOUGH_DATA"):
        tune_arima(_df(500))
