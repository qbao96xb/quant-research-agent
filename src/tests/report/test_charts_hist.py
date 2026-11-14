import json

import numpy as np
import pandas as pd

from src.agent import charts_hist as H
from src.agent import findings as fd



def turn(*calls):
    """One user message, then one (name, args, result) tool call each."""
    out = [{"role": "user", "content": "q"}]
    for i, (name, args, data) in enumerate(calls):
        out.append({"role": "assistant", "tool_calls": [
            {"id": str(i), "function": {"name": name, "arguments": json.dumps(args)}}]})
        out.append({"role": "tool", "tool_call_id": str(i), "content": json.dumps(data)})
    return out



def test_scales_with_h():
    assert H.history_len(turn(("forecast_path", {"h": 5}, {"ok": True, "h": 5}))) == 30
    assert H.history_len(turn(("forecast_path", {"h": 10}, {"ok": True, "h": 10}))) == 60
    assert H.history_len(turn(("forecast", {"h": 2}, {"ok": True}))) == 30



def test_default_without_forecast():
    assert H.history_len(turn(("load_data", {"ticker": "KO"}, {"ok": True}))) == 60



def test_latest_forecast_wins():
    msgs = turn(("forecast_path", {"h": 5}, {"ok": True, "h": 5}),
                ("forecast_path", {"h": 20}, {"ok": True, "h": 20}))
    assert H.history_len(msgs) == 120



def test_write_findings_uses_scaled_history(tmp_path, monkeypatch):
    prices = pd.DataFrame({"timestamps": pd.bdate_range("2025-01-01", periods=300),
                           "close": 100 * np.exp(np.linspace(0, 0.1, 300))})
    monkeypatch.setattr(fd, "ROOT", tmp_path)
    monkeypatch.setattr(fd, "_load_dataset", lambda _id: prices)
    msgs = turn(("load_data", {"ticker": "KO"}, {"ok": True, "dataset_id": "ds_1"}),
                ("forecast_path", {"dataset_id": "ds_1", "h": 5}, {"ok": True, "h": 5}))
    fd.write_findings(dict(run_id="hist1", status="ok", answer="done", messages=msgs), "forecast KO")
    assert len(fd.CTX["history"]["KO"]) == 30
