import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agent import charts_v2 as V
from src.agent import charts_vol as C
from src.agent import findings as fd

HIST = [{"date": str(d.date()), "cond_vol": 0.012} for d in pd.bdate_range("2026-08-03", periods=60)]


def vol_df(h=5):
    d = np.arange(1, h + 1)
    daily = 0.01 + 0.0002 * d
    return pd.DataFrame(dict(ticker="KO", day=d, daily_sigma=daily,
                             sigma=np.sqrt(np.cumsum(daily ** 2)), lo=-0.02, hi=0.02))


def test_sigma_note_numbers():
    note = C.sigma_note("KO", 0.01, 60.0)
    assert "1.00%" in note and "$60.00" in note
    assert f"${60 * np.exp(-0.01):.2f}" in note and f"${60 * np.exp(0.01):.2f}" in note
    assert "68%" in C.sigma_note("KO", 0.01)


def test_cond_vol_from_messages():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "garch_volatility", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1",
             "content": json.dumps({"ok": True, "ticker": "KO", "model": "GARCH-1-0-1-t", "history": HIST})}]
    out = C.cond_vol(msgs)
    d, model = out["KO"]
    assert len(d) == 60 and model == "GARCH-1-0-1-t"


def test_vol_cond_draws_gray_history_red_forecast_and_note(monkeypatch):
    hist = pd.DataFrame(HIST).assign(date=lambda x: pd.to_datetime(x["date"]))
    prices = pd.DataFrame({"timestamps": hist["date"], "close": 60.0})
    monkeypatch.setitem(fd.CTX, "cond_vol", {"KO": (hist, "GARCH-1-0-1-t")})
    monkeypatch.setitem(fd.CTX, "history", {"KO": prices})
    df = fd.coerce(vol_df())
    fig = C.vol_cond(df, fd.vol_spec(df))
    ax = fig.axes[0]
    colors = [l.get_color() for l in ax.get_lines()]
    assert "gray" in colors and "tab:red" in colors
    assert any("±1 sigma is about $59.39 to $60.62" in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_vol_cond_skips_without_history(monkeypatch):
    monkeypatch.setitem(fd.CTX, "cond_vol", {})
    df = fd.coerce(vol_df())
    assert C.vol_cond(df, fd.vol_spec(df)) is None


def test_vol_plan_registered():
    assert fd.CHARTS["vol"]["vol_cond"][0] is C.vol_cond
    assert V.plan("vol", "volatility for KO next 5 days", ("vol",)) == ["vol_cond", "vol_fan", "vol_daily"]
