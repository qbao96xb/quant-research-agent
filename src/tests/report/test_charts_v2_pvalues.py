import matplotlib.pyplot as plt
import pandas as pd

from src.agent import charts_v2 as V
from src.agent import findings as fd


def table(cw=True, drift=True):
    d = {"Model": ["naive", "drift", "arima"], "MAE": [0.02, 0.021, 0.022],
         "DM p-value": ["-", "0.410", "0.120"]}
    if cw:
        d["CW p-value"] = ["-", "0.030", "0.200"]
    if drift:
        d["DM p vs drift"] = ["-", "-", "0.350"]
    return fd.coerce(pd.DataFrame(d))


def test_three_groups_and_red_line():
    df = table()
    fig = V.pvalues(df, fd.metric_spec(df))
    ax = fig.axes[0]
    assert len(ax.containers) == 3
    assert [t.get_text() for t in ax.get_xticklabels()] == ["drift", "arima"]
    assert any(list(l.get_ydata()) == [0.05, 0.05] for l in ax.get_lines())
    plt.close(fig)


def test_dm_only_table_still_works():
    df = table(cw=False, drift=False)
    fig = V.pvalues(df, fd.metric_spec(df))
    assert len(fig.axes[0].containers) == 1
    plt.close(fig)


def test_no_p_columns_returns_none():
    df = fd.coerce(pd.DataFrame({"Model": ["naive", "arima"], "MAE": [0.02, 0.03]}))
    assert V.pvalues(df, fd.metric_spec(df)) is None


def test_registered_with_cw_trigger():
    fn, words = fd.CHARTS["metrics"]["pvalues"]
    assert fn is V.pvalues
    assert "cw" in words
    assert "pvalues" in fd.pick_kinds("metrics", "show the CW result")
