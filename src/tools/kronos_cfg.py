# src/tools/kronos_cfg.py
import json
from .config import KRONOS_LOOKBACK, KRONOS_T


def get_kronos_cfg() -> dict:
    """Tuned config if the tuner chose one, else the defaults from config.py."""
    from .cache import STORE            # lazy import avoids circular imports
    cfg = dict(lookback=KRONOS_LOOKBACK, T=KRONOS_T)
    p = STORE / "tuned_config.json"
    if p.exists():
        chosen = json.loads(p.read_text()).get("chosen")
        if isinstance(chosen, dict):
            cfg.update(lookback=int(chosen["lookback"]), T=float(chosen["T"]))
    return cfg
