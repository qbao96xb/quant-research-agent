import json

from src.agent import charts_v2 as V
from src.agent import findings as fd


def turn(*names):
    """One user message followed by one tool call and result per name."""
    out = [{"role": "user", "content": "q"}]
    for i, name in enumerate(names):
        out.append({"role": "assistant", "tool_calls": [
            {"id": f"{name}{i}", "function": {"name": name, "arguments": "{}"}}]})
        out.append({"role": "tool", "tool_call_id": f"{name}{i}", "content": json.dumps({"ok": True})})
    return out



def test_intents_in_priority_order():
    assert V.intents(turn("forecast_path", "tune_arima")) == ("tune", "forecast")
    assert V.intents(turn("run_backtest", "compare_models")) == ("compare",)
    assert V.intents(turn("load_data")) == ()



def test_intents_ignore_earlier_turns():
    msgs = turn("compare_models") + turn("forecast")
    assert V.intents(msgs) == ("forecast",)



def test_compare_plan_leads_with_pvalues():
    assert V.plan("metrics", "Compare ARIMA against naive for KO", ("compare",)) == ["pvalues", "scatter", "bar"]
    assert V.plan("metrics", "Compare ARIMA vs naive for KO", ("compare",)) == ["pvalues", "scatter", "bar", "rel_rmse"]



def test_trigger_words_add_charts_and_cap_at_four():
    assert V.plan("metrics", "show a heatmap", ("compare",)) == ["pvalues", "scatter", "bar", "heatmap"]
    assert V.plan("metrics", "bar dm relative coverage heatmap", ("compare",)) == ["pvalues", "scatter", "bar", "dm"]



def test_forecast_plan_and_fallbacks():
    assert V.plan("path", "forecast KO", ("forecast",)) == ["price_fan", "daily_bar"]
    assert V.plan("path", "forecast KO", ()) == fd.pick_kinds("path", "forecast KO")
    assert V.plan("vol", "volatility", ("forecast",)) == fd.pick_kinds("vol", "volatility")
