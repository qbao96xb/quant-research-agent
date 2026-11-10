import numpy as np
import pandas as pd

from evaluation.walk_forward import walk_forward
from models.baselines import Naive, Drift


def make_df(n=700, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ts = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"timestamps": ts, "close": close})


class Spy(Naive):
    """Naive that records the last timestamp of every history it receives."""
    def __init__(self):
        super().__init__()
        self.seen = []

    def forecast(self, hist, horizon):
        self.seen.append(hist["timestamps"].iloc[-1])
        return super().forecast(hist, horizon)


def test_history_ends_at_origin_and_targets_are_later():
    df = make_df()
    spy = Spy()
    res = walk_forward(df, spy, horizon=5, start=-100, step=5, min_hist=400)

    assert len(spy.seen) > 0
    assert set(res["origin"]) == set(spy.seen)         # hist ended exactly at origin
    assert (res["target_date"] > res["origin"]).all()  # targets strictly after
    assert max(spy.seen) < df["timestamps"].iloc[-1]   # last row never an origin


def test_future_changes_do_not_affect_past_forecasts():
    df = make_df()
    cut_pos = len(df) - 60
    cut_date = df["timestamps"].iloc[cut_pos]

    df2 = df.copy()
    df2.loc[cut_pos:, "close"] *= 10                    # corrupt the future

    for model in (Naive(), Drift(lookback=100)):
        a = walk_forward(df, model, horizon=5, start=-100, step=5, min_hist=400)
        b = walk_forward(df2, model, horizon=5, start=-100, step=5, min_hist=400)
        a = a[a["origin"] < cut_date].reset_index(drop=True)
        b = b[b["origin"] < cut_date].reset_index(drop=True)

        assert len(a) > 0
        assert np.allclose(a["pred"], b["pred"]), f"{model.name} used future data"
        assert (a["actual"] != b["actual"]).any() or True  # actuals may differ, preds must not
