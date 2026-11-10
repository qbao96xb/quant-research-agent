from dataclasses import dataclass
import pandas as pd

@dataclass
class Forecast:
    index: pd.DatetimeIndex   # forecast dates
    mean: pd.Series           # point forecast of close
    lower: pd.Series | None = None
    upper: pd.Series | None = None

class Forecaster:
    name: str = "base"
    def forecast(self, hist: pd.DataFrame, horizon: int) -> Forecast:
        raise NotImplementedError

# import numpy as np
# import pandas as pd
# import torch

# # (sys.path block from section 2 goes here, before this import)
# from .base import Forecaster, Forecast
# from model import Kronos, KronosTokenizer, KronosPredictor
#
# COLS = ["open", "high", "low", "close", "volume", "amount"]
#
# class KronosForecaster(Forecaster):
#     def __init__(self, model_name="NeoQuasar/Kronos-small",
#                  tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
#                  max_context=512, lookback=400, n_samples=20,
#                  T=1.0, top_p=0.9, seed=42, device=None):
#         self.name = f"kronos_{model_name.split('/')[-1]}"
#         self.lookback, self.n_samples = lookback, n_samples
#         self.T, self.top_p, self.seed = T, top_p, seed
#         self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
#         tok = KronosTokenizer.from_pretrained(tokenizer_name)
#         mdl = Kronos.from_pretrained(model_name)
#         tok.eval(); mdl.eval()
#         self.predictor = KronosPredictor(mdl, tok, device=self.device, max_context=max_context)
#
#     def forecast(self, hist: pd.DataFrame, horizon: int) -> Forecast:
#         h = hist.iloc[-self.lookback:].reset_index(drop=True)
#         x_ts = pd.Series(pd.to_datetime(h["timestamps"]))
#         y_ts = pd.Series(pd.bdate_range(x_ts.iloc[-1] + pd.Timedelta(days=1), periods=horizon))
#
#         torch.manual_seed(self.seed)
#         paths = []
#         for _ in range(self.n_samples):
#             p = self.predictor.predict(
#                 df=h[COLS], x_timestamp=x_ts, y_timestamp=y_ts,
#                 pred_len=horizon, T=self.T, top_p=self.top_p, sample_count=1,
#                 verbose=False,
#             )
#             paths.append(p["close"].to_numpy())
#         arr = np.vstack(paths)                       # (n_samples, horizon)
#         idx = pd.DatetimeIndex(y_ts)
#         return Forecast(
#             index=idx,
#             mean=pd.Series(arr.mean(0), index=idx),
#             lower=pd.Series(np.quantile(arr, 0.05, axis=0), index=idx),
#             upper=pd.Series(np.quantile(arr, 0.95, axis=0), index=idx),
#         )
