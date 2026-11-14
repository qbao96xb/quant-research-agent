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

def test_default_cfg_is_forecast_cfg():
    from tools.config import ARIMA_CFG
    out = tune_arima(_df(), max_p=1, max_q=1)
    assert out["cfg"] == ARIMA_CFG

def test_custom_cfg_is_used():
    fast = dict(lookback=750, refit_every=10)
    out = tune_arima(_df(), max_p=1, max_q=1, cfg=fast)
    assert out["cfg"] == fast and out["orders_tried"] == 4


def _d(p, verdict):
    return {"p": p, "dm": 0.0, "verdict": verdict, "n": 40}


def test_check_pass():
    from evaluation.tuning import held_out_check
    s, r = held_out_check(_d(0.30, "no difference"), _d(0.01, "A better"))
    assert s == "pass" and r


def test_check_fail_vs_naive():
    from evaluation.tuning import held_out_check
    s, r = held_out_check(_d(0.013, "A worse"), None)
    assert s == "fail" and "naive" in r and "0.013" in r


def test_check_fail_vs_default():
    from evaluation.tuning import held_out_check
    s, r = held_out_check(_d(0.40, "no difference"), _d(0.015, "A worse"))
    assert s == "fail" and "ARIMA(1,0,1)" in r and "naive" not in r


def test_output_has_check_fields():
    out = tune_arima(_df(), max_p=1, max_q=1, cfg=dict(lookback=750, refit_every=10))
    assert out["held_out_check"] in ("pass", "fail") and out["check_reason"]

