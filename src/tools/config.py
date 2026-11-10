# src/tools/config.py
"""Single source of truth for settings shared across tools and the cache key."""
from models.volatility import CachedVol

# Volatility spec: (vol, p, o, q, dist). Used for every model's interval.
VOL_SPEC = dict(vol="GARCH", p=1, o=0, q=1, dist="t")
VOL_NAME = "GARCH-1-0-1-t"   # label for the cache key

KRONOS_SAMPLES = 5
KRONOS_LOOKBACK = 120
KRONOS_T = 0.3   # manual choice: best validation row at lookback 120
MIN_HIST = 500               # must match walk_forward's min_hist


def make_vol(cached: bool = False):
    """Build the volatility model from VOL_SPEC.

    cached=True: CachedVol, fits once per (origin, horizon). Use in backtests.
    cached=False: plain VolModel. Use for single forecasts.
    """
    from models.volatility import VolModel
    return CachedVol(**VOL_SPEC) if cached else VolModel(**VOL_SPEC)
