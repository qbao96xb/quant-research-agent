import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.agent import charts_v2 as V
from src.agent import findings as fd


@pytest.mark.parametrize("n, rc", [(1, (1, 1)), (2, (1, 2)), (3, (1, 3)), (4, (2, 2)),
                                   (5, (2, 3)), (6, (2, 3)), (7, (3, 3))])
def test_grid(n, rc):
    assert V.grid(n) == rc



def metrics_df():
    return pd.DataFrame(dict(model=["naive", "drift", "arima"],
                             MAE=[0.020, 0.021, 0.022], RMSE=[0.030, 0.031, 0.029],
                             DirAcc=[0.50, 0.52, 0.48], Cov90=[0.90, 0.88, 0.91],
                             Width=[0.08, 0.09, 0.08]))


def test_metric_grid_is_2x3_with_one_hidden():
    df = metrics_df()
    fig = V.metric_grid(df, fd.metric_spec(df))
    assert len(fig.axes) == 6
    assert sum(ax.get_visible() for ax in fig.axes) == 5
    plt.close(fig)


def test_metric_grid_needs_metrics():
    df = pd.DataFrame(dict(model=["naive"], x=[1.0]))
    assert V.metric_grid(df, fd.metric_spec(df)) is None


def test_bar_entry_uses_metric_grid():
    assert fd.CHARTS["metrics"]["bar"][0] is V.metric_grid
