# src/tools/model_names.py
from pathlib import Path

_FT = (Path(__file__).resolve().parents[2]
       / "outputs" / "kronos_ft_small_listc" / "basemodel" / "best_model")
KRONOS_SIZES = ("mini", "small", "base") + (("small_ft",) if _FT.exists() else ())
BASELINES = ("naive", "drift", "arima")
MODEL_NAMES = BASELINES + tuple(f"kronos_{s}" for s in KRONOS_SIZES)
