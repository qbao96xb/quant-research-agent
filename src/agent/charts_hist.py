"""Price-history length for the forecast fan chart.

findings.py imports this module as `H`.
"""
MIN_DAYS = 30        # shortest history shown
PER_STEP = 6         # history days per forecast day
DEFAULT_DAYS = 60    # when the conversation has no forecast


def history_len(messages):
    """Trading days of history: max(MIN_DAYS, PER_STEP * h) for the latest forecast h, else DEFAULT_DAYS."""
    from . import findings as F
    h = None
    for _, name, args, data in F._calls(messages):
        if name not in ("forecast", "forecast_path"):
            continue
        value = data.get("h") if isinstance(data, dict) else None
        if value is None:
            value = (args or {}).get("h")
        try:
            h = int(value)
        except (TypeError, ValueError):
            continue
    return DEFAULT_DAYS if h is None else max(MIN_DAYS, PER_STEP * h)
