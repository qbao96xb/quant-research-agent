"""fit_json keeps long tool results valid JSON and keeps the summary fields (C02, I02, S02)."""
import json

from src.agent.fit_output import fit_json


def test_short_result_unchanged():
    r = {"ok": True, "x": 1}
    assert fit_json(r) == json.dumps(r)


def test_long_rows_keep_summary():
    rows = [{"origin": "2026-01-01", "pred_return": 0.0123456789012345, "i": i} for i in range(300)]
    r = {"ok": True, "rows": rows, "summary": {"hit_rate": 0.9, "rmse": 0.0123}}
    s = fit_json(r)
    assert len(s) <= 8000
    out = json.loads(s)
    assert out["summary"] == r["summary"]
    assert 0 < len(out["rows"]) < 300
    assert out["rows"] == rows[: len(out["rows"])]
    assert out["truncated"]["rows"] == "showing the first " + str(len(out["rows"])) + " of 300"


def test_forecast_path_20_days_kept_whole():
    day = {"ticker": "KO", "model": "arima", "price": 95.7232, "daily_pct": "+0.06%", "cum_pct": "+0.06%",
           "cum_lo_pct": "-1.87%", "cum_hi_pct": "+1.99%", "cum_log_return": 0.000556093,
           "daily_log_return": 0.000556093, "lo": -0.0187402, "hi": 0.0198523}
    path = [dict(day, day=i + 1) for i in range(20)]
    r = {"ok": True, "h": 20, "path": path, "last_close": 95.67}
    s = fit_json(r)
    assert len(json.dumps(path)) > 4000
    assert json.loads(s) == r


def test_artifact_list_keeps_note():
    arts = [{"id": "bt_" + str(i).zfill(10), "created": "2026-11-14 10:00"} for i in range(300)]
    r = {"ok": True, "artifacts": arts, "datasets": 12, "note": "Pick the most recent dataset."}
    out = json.loads(fit_json(r))
    assert out["note"] == r["note"]
    assert out["datasets"] == 12
    assert 0 < len(out["artifacts"]) < 300
    assert out["artifacts"] == arts[: len(out["artifacts"]) ]


def test_input_not_changed():
    r = {"rows": list(range(5000))}
    fit_json(r)
    assert len(r["rows"]) == 5000


def test_long_text_without_lists_is_still_cut():
    r = {"ok": True, "text": "x" * 20000}
    assert len(fit_json(r)) < 20000
