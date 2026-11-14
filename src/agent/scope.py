"""Per-request call caps for slow tools (owner, 2026-11-13). A new user message starts a new request."""
CAPPED_TOOLS = {"tune_arima", "make_report", "run_backtest", "forecast_vs_actual", "tune_kronos"}
CALL_CAP = 1
CAP_MSG = ("{name} already ran in this request. Answer with the result you have, "
           "or ask the user before running it again.")

def cap_block(name, done):
    """(result, meta) refusing this call when the tool reached its cap in this request, else None."""
    if name in CAPPED_TOOLS and done.get(name, 0) >= CALL_CAP:
        result = {"ok": False, "error": {"code": "CALL_CAP", "message": CAP_MSG.format(name=name)}}
        return result, {"ok": False, "error": "CALL_CAP", "blocked": True}
    return None

def count_call(name, result, meta, done):
    """Count a call that did work (succeeded or timed out). Errors, cached and blocked calls do not count."""
    if meta.get("cached") or meta.get("blocked"):
        return
    code = (result.get("error") or {}).get("code") if isinstance(result, dict) else None
    if meta.get("ok") or code == "TIMEOUT":
        done[name] = done.get(name, 0) + 1
