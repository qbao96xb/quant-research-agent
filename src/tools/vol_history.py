"""Recent conditional volatility for garch_volatility (in-sample, no look-ahead)."""
from .config import make_vol

HISTORY_DAYS = 60


def vol_history(hist, n=HISTORY_DAYS):
    """Last n days of in-sample conditional daily sigma (decimal) from the configured vol model.

    Uses only the rows of `hist`, so every date is at or before the forecast origin.
    Returns [] if the fit fails, so the forecast itself is never blocked.
    """
    try:
        d = make_vol().diagnostics(hist, last=n)
    except Exception:
        return []
    return [dict(date=t, cond_vol=float(v)) for t, v in zip(d["dates"], d["cond_vol"]) ]