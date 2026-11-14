import matplotlib.pyplot as plt
import pandas as pd

from src.agent import charts_v2 as V
from src.agent import findings as fd

def test_ticker_and_model_columns_label_both():
    df = fd.coerce(pd.DataFrame({"ticker": ["KO", "KO", "KO"],
                                 "Model": ["naive", "drift", "arima"],
                                 "DM p-value": ["-", "0.410", "0.120"],
                                 "CW p-value": ["-", "0.030", "0.200"]}))
    fig = V.pvalues(df, fd.metric_spec(df))
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["drift (KO)", "arima (KO)"]
    plt.close(fig)
