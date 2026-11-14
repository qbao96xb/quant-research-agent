import json

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from src.agent import charts_v2 as V
from src.agent import findings as fd

TABLE = "\n".join(["| Order | Validation RMSE | Fallbacks |", "|---|---|---|",
                   "| (2, 0, 1) | 0.029900 | 0 |",
                   "| (1, 0, 1) | 0.030100 | 0 |",
                   "| (0, 0, 1) | 0.030500 | 1 |"])


def tune_df():
    return fd.coerce(next(fd.pipe_tables(TABLE)))


def test_kind_of_tune_table_only():
    assert V.kind_of(tune_df()) == "tune"
    assert V.kind_of(pd.DataFrame({"model": ["arima"], "RMSE": [0.03]})) is None


def test_tune_rmse_highlights_best_order():
    df = tune_df()
    fig = V.tune_rmse(df, V.tune_spec(df))
    bars = fig.axes[0].patches
    assert len(bars) == 3
    green = [b for b in bars if b.get_facecolor() == mcolors.to_rgba("tab:green")]
    assert len(green) == 1 and abs(green[0].get_width() - 0.0299) < 1e-9
    plt.close(fig)


def test_tune_kind_registered():
    assert fd.SPEC["tune"] is V.tune_spec
    assert fd.DEFAULT["tune"] == ["tune_rmse"]
    assert fd.LABEL["tune"] == "ARIMA order search"


def test_write_findings_draws_tune_chart(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "ROOT", tmp_path)
    msgs = [{"role": "user", "content": "tune ARIMA for KO"},
            {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "tune_arima", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1",
             "content": json.dumps({"ok": True, "table": TABLE, "best_order": [2, 0, 1]})}]
    md, _, charts = fd.write_findings(dict(run_id="tune1", status="ok", answer="done", messages=msgs),
                                      "tune ARIMA for KO")
    assert any("tune_rmse" in p.name for p in charts)

