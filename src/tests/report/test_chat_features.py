"""Tests for the chat-side forecast charts: forecast_vs_actual tool, track-record charts,
fan chart with history, daily volatility bars, default chart sets and gallery layout."""
import json
from pathlib import Path

import matplotlib
import matplotlib.collections, matplotlib.dates
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from src.agent import findings as fd
from src.tools import track_tools as tt
from src.tools.registry import REGISTRY, call
from models.baselines import Naive, Drift

ROOT = Path(__file__).resolve().parents[3]


# ---------- fixtures ----------
@pytest.fixture
def prices():
    rng = np.random.default_rng(1)
    n = 900
    return pd.DataFrame({"timestamps": pd.bdate_range("2021-01-01", periods=n),
                         "close": 70 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))})


class FakeVol:
    """Constant sigma: keeps the tool tests fast and exact."""
    name = "fake"
    def sigma(self, hist, horizon):
        return 0.01 * np.sqrt(np.arange(1, horizon + 1))
    def reset(self):
        pass


@pytest.fixture
def track_rows(prices):
    return tt.compute_track(prices, Naive(), n=30, h=1, vol=FakeVol())


@pytest.fixture
def track_df(track_rows):
    return pd.DataFrame(track_rows["rows"]).assign(ticker="KO", model="naive")


def _track_frame(n, seed=0, bias=0.0):
    rng = np.random.default_rng(seed)
    pred = rng.normal(0, 0.003, n)
    act = pred + rng.normal(bias, 0.01, n)
    return pd.DataFrame(dict(
        ticker="KO", model="arima", origin=pd.bdate_range("2026-01-01", periods=n).strftime("%Y-%m-%d"),
        date=pd.bdate_range("2026-01-02", periods=n).strftime("%Y-%m-%d"),
        pred_return=pred, actual_return=act, lo=pred - 0.016, hi=pred + 0.016,
        error=pred - act, inside=np.abs(act - pred) <= 0.016))


def _path_df(h=15, ticker="KO"):
    d = np.arange(1, h + 1)
    return pd.DataFrame(dict(ticker=ticker, model="arima", day=d, price=70 * np.exp(0.0005 * d),
                             cum_log_return=0.0005 * d, daily_log_return=0.0005,
                             lo=-0.015 * np.sqrt(d), hi=0.015 * np.sqrt(d)))


def _vol_df(h=15):
    d = np.arange(1, h + 1)
    daily = 0.01 + 0.0002 * d
    return pd.DataFrame(dict(ticker="KO", day=d, daily_sigma=daily, sigma=np.sqrt(np.cumsum(daily ** 2)),
                             lo=-0.02, hi=0.02))


# =====================================================================
# 1. compute_track: the numbers
# =====================================================================
def test_track_has_n_rows_and_columns(track_rows):
    rows = track_rows["rows"]
    assert len(rows) == 30 == track_rows["summary"]["n"]
    assert set(rows[0]) >= {"origin", "date", "pred_return", "actual_return", "lo", "hi", "error", "inside"}


def test_track_naive_predicts_zero(track_rows):
    assert all(abs(r["pred_return"]) < 1e-12 for r in track_rows["rows"])


def test_track_actual_return_matches_prices(prices, track_rows):
    px = prices.set_index(prices["timestamps"].dt.strftime("%Y-%m-%d"))["close"]
    for r in track_rows["rows"][:5] + track_rows["rows"][-5:]:
        assert r["actual_return"] == pytest.approx(np.log(px[r["date"]] / px[r["origin"]]), abs=1e-5)


def test_track_covers_the_most_recent_days(prices, track_rows):
    assert track_rows["rows"][-1]["date"] == str(prices["timestamps"].iloc[-1])[:10]
    dates = [r["date"] for r in track_rows["rows"]]
    assert dates == sorted(dates) and len(set(dates)) == len(dates)


def test_track_error_and_inside_are_consistent(track_rows):
    for r in track_rows["rows"]:
        assert r["error"] == pytest.approx(r["pred_return"] - r["actual_return"], abs=1e-5)
        assert r["inside"] == (r["lo"] <= r["actual_return"] <= r["hi"])
        assert r["lo"] < r["pred_return"] < r["hi"]


def test_track_summary_matches_rows(track_rows):
    rows, s = track_rows["rows"], track_rows["summary"]
    assert s["hit_rate"] == pytest.approx(np.mean([r["inside"] for r in rows]), abs=1e-3)
    err = np.array([r["error"] for r in rows])
    assert s["rmse"] == pytest.approx(np.sqrt((err ** 2).mean()), abs=1e-5)
    assert s["mae"] == pytest.approx(np.abs(err).mean(), abs=1e-5)
    assert s["bias"] == pytest.approx(err.mean(), abs=1e-5)


def test_track_dir_acc_is_none_for_constant_zero_forecast(track_rows):
    assert track_rows["summary"]["dir_acc"] is None


def test_track_dir_acc_defined_for_drift(prices):
    out = tt.compute_track(prices, Drift(lookback=250), n=30, h=1, vol=FakeVol())
    assert 0.0 <= out["summary"]["dir_acc"] <= 1.0


def test_track_multi_horizon_rows_use_h_day_returns(prices):
    out = tt.compute_track(prices, Naive(), n=10, h=3, vol=FakeVol())
    r = out["rows"][-1]
    px = prices.set_index(prices["timestamps"].dt.strftime("%Y-%m-%d"))["close"]
    assert len(out["rows"]) == 10
    assert r["actual_return"] == pytest.approx(np.log(px[r["date"]] / px[r["origin"]]), abs=1e-5)


# ---------- no look-ahead ----------
def test_track_no_lookahead(prices):
    seen = []

    class Spy(Naive):
        def forecast(self, hist, h):
            seen.append(hist["timestamps"].iloc[-1])
            return super().forecast(hist, h)

    out = tt.compute_track(prices, Spy(), n=12, h=2, vol=FakeVol())
    assert len(seen) == 12
    for t, r in zip(seen, out["rows"]):
        assert str(t)[:10] == r["origin"] < r["date"]


def test_track_needs_enough_history(prices):
    with pytest.raises(Exception):
        tt.compute_track(prices.iloc[:420], Naive(), n=30, h=1, vol=FakeVol())


# =====================================================================
# 2. forecast_vs_actual tool wrapper
# =====================================================================
@pytest.fixture
def patched_tool(monkeypatch, prices):
    monkeypatch.setattr(tt, "get_dataset", lambda _id: prices)
    monkeypatch.setattr(tt, "make_model", lambda name, order=None: Naive())
    monkeypatch.setattr(tt, "make_vol", lambda cached=False: FakeVol())
    monkeypatch.setattr(tt, "get_meta", lambda _id: {"ticker": "KO"})


def test_tool_registered_and_documented():
    assert "forecast_vs_actual" in REGISTRY
    assert REGISTRY["forecast_vs_actual"]["fn"].__doc__


def test_tool_returns_rows_and_summary(patched_tool):
    out = call("forecast_vs_actual", dataset_id="ds_x", model="naive", n=20, h=1)
    assert out["ok"] and out["ticker"] == "KO" and out["model"] == "naive"
    assert len(out["rows"]) == 20 and out["summary"]["n"] == 20 and out["level"] == "90%"


def test_tool_rows_are_json_serialisable(patched_tool):
    json.dumps(call("forecast_vs_actual", dataset_id="ds_x", model="naive", n=10))


@pytest.mark.parametrize("n", [0, 4, 121])
def test_tool_rejects_bad_n(n):
    out = call("forecast_vs_actual", dataset_id="ds_x", model="naive", n=n)
    assert out["ok"] is False


def test_tool_rejects_order_for_non_arima():
    out = call("forecast_vs_actual", dataset_id="ds_x", model="naive", order=[1, 0, 1])
    assert out["ok"] is False


def test_tool_unknown_dataset_is_clean_error():
    out = call("forecast_vs_actual", dataset_id="ds_does_not_exist", model="naive")
    assert out["ok"] is False and "error" in out


# =====================================================================
# 3. classification: a track table is its own kind
# =====================================================================
def test_kind_of_track_table(track_df):
    assert fd.kind_of(track_df) == "track"


def test_kind_of_path_table_unchanged():
    assert fd.kind_of(_path_df()) == "path"


def test_kind_of_vol_table_unchanged():
    assert fd.kind_of(_vol_df()) == "vol"


def test_track_table_is_not_confused_with_errors(track_df):
    assert fd.kind_of(track_df.assign(model="naive")) == "track"


# =====================================================================
# 4. track-record charts
# =====================================================================
def _spec(df):
    return fd.SPEC["track"](fd.coerce(df))


def test_track_bars_draws_pairs_and_marks_breaches():
    df = _track_frame(30)
    fig = fd.track_bars(fd.coerce(df), _spec(df))
    ax = fig.axes[0]
    bars = [p for p in ax.patches]
    assert len(bars) == 60                                   # predicted + actual per day
    breaches = int((~df["inside"]).sum())
    red = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)
           and len(c.get_offsets()) and np.allclose(c.get_facecolor()[0][:3], matplotlib.colors.to_rgb("tab:red"))]
    assert sum(len(c.get_offsets()) for c in red) == breaches
    assert f"{df['inside'].mean():.0%}" in ax.get_title()


def test_track_bars_one_panel_per_model():
    df = pd.concat([_track_frame(20, 0), _track_frame(20, 1).assign(model="drift")])
    fig = fd.track_bars(fd.coerce(df), _spec(df))
    assert len([a for a in fig.axes if a.get_visible()]) == 2


def test_track_bars_missing_columns_returns_none():
    df = _track_frame(10).drop(columns=["actual_return"])
    assert fd.track_bars(fd.coerce(df), _spec(df)) is None


def test_track_scatter_needs_20_points():
    small = _track_frame(10)
    assert fd.track_scatter(fd.coerce(small), _spec(small)) is None


def test_track_scatter_reports_slope_and_r2():
    df = _track_frame(40)
    fig = fd.track_scatter(fd.coerce(df), _spec(df))
    title = fig.axes[0].get_title()
    assert "slope" in title and "R" in title
    assert len(fig.axes[0].collections[0].get_offsets()) == 40


def test_track_scatter_has_45_degree_reference():
    df = _track_frame(40)
    ax = fd.track_scatter(fd.coerce(df), _spec(df)).axes[0]
    ref = [l for l in ax.lines if l.get_linestyle() == "--"]
    assert ref and np.allclose(ref[0].get_xdata(), ref[0].get_ydata())


# =====================================================================
# 5. price fan with history, daily volatility bars
# =====================================================================
@pytest.fixture
def ctx(prices):
    fd.CTX["history"] = {"KO": prices.tail(120)}
    fd.CTX["level"] = 0.90
    yield fd.CTX
    fd.CTX["history"] = {}
    fd.CTX["level"] = 0.90


def _pspec(df):
    return fd.SPEC["path"](fd.coerce(df))


def test_price_fan_includes_history_and_bands(ctx):
    df = _path_df(15)
    fig = fd.price_fan(fd.coerce(df), _pspec(df))
    ax = fig.axes[0]
    longest = max(len(l.get_xdata()) for l in ax.lines)
    assert longest >= 60                                      # history line is drawn
    assert len(ax.collections) >= 2                           # nested bands


def test_price_fan_forecast_starts_at_last_close(ctx):
    df = _path_df(15)
    ax = fd.price_fan(fd.coerce(df), _pspec(df)).axes[0]
    last = ctx["history"]["KO"]["close"].iloc[-1]
    fc = [l for l in ax.lines if len(l.get_xdata()) == 16][0]  # anchor + 15 days
    assert fc.get_ydata()[0] == pytest.approx(last)


def test_price_fan_bands_widen_with_horizon(ctx):
    df = _path_df(15)
    ax = fd.price_fan(fd.coerce(df), _pspec(df)).axes[0]
    verts = ax.collections[0].get_paths()[0].vertices
    ys = verts[:, 1]
    assert ys.max() - ys.min() > 0


def test_price_fan_dates_are_business_days_after_last_history(ctx):
    df = _path_df(15)
    ax = fd.price_fan(fd.coerce(df), _pspec(df)).axes[0]
    fc = [l for l in ax.lines if len(l.get_xdata()) == 16][0]
    x = pd.to_datetime(matplotlib.dates.num2date(fc.get_xdata(orig=False)))
    assert x[0] == ctx["history"]["KO"]["timestamps"].iloc[-1].tz_localize(x.tz) if x.tz else True
    assert all(d.weekday() < 5 for d in x[1:])


def test_price_fan_without_history_falls_back_to_price_band():
    fd.CTX["history"] = {}
    df = _path_df(15)
    assert fd.price_fan(fd.coerce(df), _pspec(df)) is not None


def test_price_fan_multiple_tickers(ctx, prices):
    ctx["history"]["PEP"] = prices.tail(120)
    df = pd.concat([_path_df(15, "KO"), _path_df(15, "PEP")])
    fig = fd.price_fan(fd.coerce(df), _pspec(df))
    assert len([a for a in fig.axes if a.get_visible()]) == 2


def test_vol_daily_bar_heights_match_daily_sigma():
    df = _vol_df(15)
    fig = fd.vol_daily(fd.coerce(df), fd.SPEC["vol"](fd.coerce(df)))
    heights = sorted(p.get_height() for p in fig.axes[0].patches)
    assert heights == pytest.approx(sorted(df["daily_sigma"] * 100), rel=1e-6)


def test_vol_daily_needs_daily_sigma():
    df = _vol_df(15).drop(columns=["daily_sigma"])
    assert fd.vol_daily(fd.coerce(df), fd.SPEC["vol"](fd.coerce(df))) is None


# =====================================================================
# 6. defaults: several charts per table, all registered
# =====================================================================
@pytest.mark.parametrize("kind,minimum", [("path", 3), ("vol", 2), ("track", 2), ("metrics", 3), ("errors", 2)])
def test_default_sets_have_several_charts(kind, minimum):
    assert len(fd.DEFAULT[kind]) >= minimum


@pytest.mark.parametrize("kind", ["path", "vol", "track", "metrics", "errors"])
def test_every_default_is_registered(kind):
    assert all(name in fd.CHARTS[kind] for name in fd.DEFAULT[kind])


def test_path_default_leads_with_fan():
    assert fd.DEFAULT["path"][0] == "price_fan"


def test_trigger_words_still_narrow_the_set():
    assert fd.pick_kinds("path", "show me a whisker plot") == ["whisker"]


def test_pick_kinds_falls_back_to_defaults():
    assert fd.pick_kinds("path", "forecast xom") == fd.DEFAULT["path"]


# =====================================================================
# 7. context helpers
# =====================================================================
def _msgs(*calls):
    out = [{"role": "user", "content": "q"}]
    for i, (name, args, data) in enumerate(calls):
        out.append({"role": "assistant", "tool_calls": [
            {"id": str(i), "function": {"name": name, "arguments": json.dumps(args)}}]})
        out.append({"role": "tool", "tool_call_id": str(i), "content": json.dumps(data)})
    return out


def test_history_built_from_load_data(prices):
    msgs = _msgs(("load_data", {"ticker": "ko"}, {"ok": True, "dataset_id": "ds_1", "rows": 900}))
    h = fd._history(msgs, loader=lambda _id: prices, n=60)
    assert list(h) == ["KO"] and len(h["KO"]) == 60
    assert h["KO"]["close"].iloc[-1] == prices["close"].iloc[-1]


def test_history_loader_failure_is_skipped():
    def boom(_id): raise RuntimeError("gone")
    msgs = _msgs(("load_data", {"ticker": "KO"}, {"ok": True, "dataset_id": "ds_1"}))
    assert fd._history(msgs, loader=boom) == {}


def test_level_parsed_from_forecast_path():
    msgs = _msgs(("forecast_path", {"dataset_id": "ds_1"}, {"ok": True, "level": "95%", "path": []}))
    assert fd._level(msgs) == pytest.approx(0.95)


def test_level_defaults_to_90():
    assert fd._level(_msgs()) == pytest.approx(0.90)


# =====================================================================
# 8. end to end: one turn produces many charts
# =====================================================================
def test_write_findings_produces_charts_for_every_kind(tmp_path, monkeypatch, prices, track_rows):
    monkeypatch.setattr(fd, "ROOT", tmp_path)
    monkeypatch.setattr(fd, "_load_dataset", lambda _id: prices)
    path = _path_df(15).drop(columns=["ticker", "model"]).to_dict("records")
    vol = _vol_df(15).drop(columns=["ticker"]).to_dict("records")
    msgs = _msgs(
        ("load_data", {"ticker": "KO"}, {"ok": True, "dataset_id": "ds_1"}),
        ("forecast_path", {"dataset_id": "ds_1", "model": "arima", "h": 15},
         {"ok": True, "model": "arima", "level": "90%", "ticker": "KO", "path": path}),
        ("garch_volatility", {"dataset_id": "ds_1", "h": 15},
         {"ok": True, "ticker": "KO", "path": [{**r, "ticker": "KO"} for r in vol]}),
        ("forecast_vs_actual", {"dataset_id": "ds_1", "model": "arima", "n": 30},
         {"ok": True, "ticker": "KO", "model": "arima", **track_rows}))
    md, _, charts = fd.write_findings(dict(run_id="e2e", status="ok", answer="done", messages=msgs),
                                      "forecast KO for the next 15 days")
    names = " ".join(p.name for p in charts)
    for expected in ("price_fan", "daily_bar", "vol_fan", "vol_daily", "track_bars"):
        assert expected in names, (expected, names)
    assert len(charts) >= 5 and all(p.exists() and p.stat().st_size > 1000 for p in charts)
    assert "failed" not in md.read_text().lower()


def test_write_findings_survives_a_failing_chart(tmp_path, monkeypatch, prices):
    monkeypatch.setattr(fd, "ROOT", tmp_path)
    monkeypatch.setattr(fd, "_load_dataset", lambda _id: prices)
    spec, orig = fd.CHARTS["path"]["price_fan"]
    fd.CHARTS["path"]["price_fan"] = (lambda df, s: 1 / 0, orig)
    try:
        path = _path_df(5).drop(columns=["ticker", "model"]).to_dict("records")
        msgs = _msgs(("forecast_path", {"dataset_id": "ds_1", "ticker": "KO"},
                      {"ok": True, "ticker": "KO", "level": "90%", "path": path}))
        md, _, charts = fd.write_findings(dict(run_id="e2e2", status="ok", answer="x", messages=msgs), "q")
        assert charts and "price_fan" in md.read_text()          # failure reported, others drawn
    finally:
        fd.CHARTS["path"]["price_fan"] = (spec, orig)


# =====================================================================
# 9. agent wiring: prompt routing and gallery layout
# =====================================================================
def test_system_prompt_routes_track_requests():
    text = (ROOT / "src/agent/prompts/system.md").read_text()
    assert "forecast_vs_actual" in text
    assert "predicted vs actual" in text.lower()


def test_system_prompt_routes_multi_day_forecasts_to_full_set():
    text = (ROOT / "src/agent/prompts/system.md").read_text().lower()
    assert "next" in text and "garch_volatility" in text and "forecast_path" in text


def test_gallery_shows_multiple_charts_at_once():
    src = (ROOT / "scripts/chat_ui.py").read_text()
    assert "gr.Gallery" in src and "preview=True" not in src
    assert "columns=1" not in src
