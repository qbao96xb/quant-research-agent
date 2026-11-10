import sys
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2]
KRONOS_DIR = ROOT / "third_party" / "Kronos"
FT_DIR = ROOT / "outputs" / "kronos_ft_small"

if str(KRONOS_DIR) not in sys.path:
    sys.path.insert(0, str(KRONOS_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402
from .base import Forecaster, Forecast  # noqa: E402

COLS = ["open", "high", "low", "close", "volume", "amount"]

KRONOS_CONFIGS = {
    "mini":  dict(model_name="NeoQuasar/Kronos-mini",
                  tokenizer_name="NeoQuasar/Kronos-Tokenizer-2k", max_context=2048),
    "small": dict(model_name="NeoQuasar/Kronos-small",
                  tokenizer_name="NeoQuasar/Kronos-Tokenizer-base", max_context=512),
    "base":  dict(model_name="NeoQuasar/Kronos-base",
                  tokenizer_name="NeoQuasar/Kronos-Tokenizer-base", max_context=512),
    "small_ft": dict(model_name=str(FT_DIR / "basemodel" / "best_model"),
                 tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
                 max_context=512),
}


class KronosForecaster(Forecaster):
    def __init__(self, model_name="NeoQuasar/Kronos-small",
                 tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
                 max_context=512, lookback=120, n_samples=20,
                 T=1.0, top_p=0.9, seed=42, device=None, label=None):
        assert lookback <= max_context, "lookback must be <= max_context"
        self.name = label or f"kronos_{model_name.split('/')[-1]}"
        self.lookback, self.n_samples = lookback, n_samples
        self.T, self.top_p, self.seed = T, top_p, seed
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        tok = KronosTokenizer.from_pretrained(tokenizer_name)
        mdl = Kronos.from_pretrained(model_name)
        tok.eval()
        mdl.eval()
        self.predictor = KronosPredictor(mdl, tok, device=self.device, max_context=max_context)
        self.config = dict(model=model_name, tokenizer=tokenizer_name, max_context=max_context,
                           lookback=lookback, n_samples=n_samples, T=T, top_p=top_p, seed=seed,
                           device=self.device, adjusted_prices=True)

    def forecast(self, hist: pd.DataFrame, horizon: int) -> Forecast:
        h = hist.iloc[-self.lookback:].reset_index(drop=True)
        x_ts = pd.Series(pd.to_datetime(h["timestamps"]))
        y_ts = pd.Series(pd.bdate_range(x_ts.iloc[-1] + pd.Timedelta(days=1), periods=horizon))

        torch.manual_seed(self.seed)
        paths = []
        for _ in range(self.n_samples):
            p = self.predictor.predict(
                df=h[COLS], x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=horizon, T=self.T, top_p=self.top_p, sample_count=1,
                verbose=False,
            )
            paths.append(p["close"].to_numpy())
        arr = np.vstack(paths).astype(float)  # (n_samples, horizon)
        last_close = float(h["close"].iloc[-1])

        if arr.shape != (self.n_samples, horizon):
            raise ValueError(f"Unexpected sample shape: {arr.shape}")

        if (
                not np.isfinite(last_close)
                or last_close <= 0
                or not np.isfinite(arr).all()
                or (arr <= 0).any()
        ):
            raise ValueError(
                "Historical and sampled prices must be finite and positive."
            )

        # Each column is a cumulative log return from the origin.
        log_returns = np.log(arr / last_close)

        mean_return = log_returns.mean(axis=0)
        lower_return = np.quantile(log_returns, 0.05, axis=0)
        upper_return = np.quantile(log_returns, 0.95, axis=0)

        idx = pd.DatetimeIndex(y_ts)

        # Keep price outputs for compatibility with Forecast and evaluation.
        return Forecast(
            index=idx,
            mean=pd.Series(last_close * np.exp(mean_return), index=idx),
            lower=pd.Series(last_close * np.exp(lower_return), index=idx),
            upper=pd.Series(last_close * np.exp(upper_return), index=idx),
        )


@lru_cache(maxsize=16)
def get_forecaster(size="small", n_samples=20, lookback=120, seed=42, T=1.0) -> KronosForecaster:
    return KronosForecaster(**KRONOS_CONFIGS[size], n_samples=n_samples,
                            lookback=lookback, seed=seed, T=T,
                            label=f"kronos_{size}_lb{lookback}_T{T}")

def kronos_forecast(
    df_hist,
    horizon=5,
    n_samples=20,
    size="small",
    lookback=400,
):
    forecaster = get_forecaster(
        size=size,
        n_samples=n_samples,
        lookback=lookback,
    )
    fc = forecaster.forecast(df_hist, horizon)
    last_close = float(df_hist["close"].iloc[-1])

    mean_return = np.log(fc.mean.to_numpy() / last_close)
    lower_return = np.log(fc.lower.to_numpy() / last_close)
    upper_return = np.log(fc.upper.to_numpy() / last_close)

    return {
        "model": forecaster.name,
        "dates": [d.strftime("%Y-%m-%d") for d in fc.index],

        # Preserve existing price-output keys.
        "mean": fc.mean.round(4).tolist(),
        "lower": fc.lower.round(4).tolist(),
        "upper": fc.upper.round(4).tolist(),

        # Unrounded cumulative log returns for evaluation.
        "log_return_mean": mean_return.tolist(),
        "log_return_lower": lower_return.tolist(),
        "log_return_upper": upper_return.tolist(),

        "end_of_horizon": {
            "horizon": horizon,
            "date": fc.index[-1].strftime("%Y-%m-%d"),
            "mean_log_return": float(mean_return[-1]),
            "lower_log_return": float(lower_return[-1]),
            "upper_log_return": float(upper_return[-1]),
        },
    }
