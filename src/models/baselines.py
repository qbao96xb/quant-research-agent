"""Return-based baselines with price outputs for the existing pipeline.

Interface:
    model.forecast(hist, horizon) -> Forecast

Required history columns:
    timestamps, close

Models:
    Naive: zero cumulative log return.
    Drift: cumulative log return = horizon * mean daily log return.
    Arima: fit daily log returns and accumulate their forecasts.

Price outputs:
    last_close * exp(predicted cumulative log return)

All models return missing interval bounds. Use
models.volatility.add_intervals to attach GARCH-based intervals.

Lookback counts DAILY RETURNS, not price observations. Estimating L
returns requires L + 1 prices. If less history is available, the models
use all available returns.

ARIMA order must be selected on pre-test data and then frozen.

Speed notes:
    - Only the last lookback + 1 rows are touched.
    - validate=False checks only the last row (walk_forward already
      validates the full price and date series).
    - Drift uses a telescoping mean: (log P_t - log P_{t-L}) / L.
    - Arima: no parameter covariance (cov_type="none"), returns rescaled
      by `scale` for better optimizer conditioning, optional refit
      schedule, warm start and shorter filtering window between refits.
"""
import warnings
from numbers import Integral, Real

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from models.base import Forecast


def _validate_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_lookback(lookback):
    _validate_positive_int("lookback", lookback)


def _prepare(hist, horizon, n_rows, validate=True):
    """Return tail closes (last n_rows), future weekdays and last timestamp.

    validate=True checks the whole tail. validate=False checks only the
    last row, assuming the caller has validated the full series.
    """
    _validate_positive_int("horizon", horizon)

    missing = {"timestamps", "close"}.difference(hist.columns)
    if missing:
        raise ValueError(f"Missing history columns: {sorted(missing)}")
    if hist.empty:
        raise ValueError("History must contain at least one observation.")

    tail = hist.iloc[-n_rows:]
    close = tail["close"].to_numpy(dtype=float)

    if validate:
        if not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError("Close prices must be finite and strictly positive.")
        ts = pd.DatetimeIndex(pd.to_datetime(tail["timestamps"]))
        if ts.hasnans or not ts.is_monotonic_increasing or ts.has_duplicates:
            raise ValueError("Timestamps must be valid, increasing, and unique.")
        last_ts = ts[-1]
    else:
        if not np.isfinite(close[-1]) or close[-1] <= 0:
            raise ValueError("Last close must be finite and strictly positive.")
        last_ts = pd.Timestamp(tail["timestamps"].iloc[-1])
        if pd.isna(last_ts):
            raise ValueError("Last timestamp is missing.")

    # Weekdays only: exchange holidays are not excluded.
    idx = pd.bdate_range(start=last_ts + pd.offsets.BDay(1), periods=horizon)
    return close, idx, last_ts


def _returns(close):
    """Daily log returns from the (already trimmed) closes."""
    if len(close) < 2:
        raise ValueError("At least two close prices are needed for returns.")
    return np.diff(np.log(close))


def _price_forecast(last_close, cumulative_returns, idx):
    """Convert cumulative log-return forecasts to prices with empty bounds."""
    values = last_close * np.exp(np.asarray(cumulative_returns, dtype=float))
    return Forecast(
        index=idx,
        mean=pd.Series(values, index=idx),
        lower=pd.Series(np.nan, index=idx),
        upper=pd.Series(np.nan, index=idx),
    )

class Naive:
    """Random walk benchmark: zero forecast log return at every horizon."""
    name = "naive"

    def __init__(self, validate=True):
        self.validate = bool(validate)
        self.config = dict(model=self.name, target="log_return")

    def forecast(self, hist, horizon):
        close, idx, _ = _prepare(hist, horizon, 1, self.validate)
        return _price_forecast(close[-1], np.zeros(horizon), idx)


class Drift:
    """Constant daily drift: mean of the last `lookback` daily log returns."""
    name = "drift"

    def __init__(self, lookback=250, validate=True):
        _validate_lookback(lookback)
        self.lookback = int(lookback)
        self.validate = bool(validate)
        self.config = dict(
            model=self.name,
            target="log_return",
            lookback=self.lookback,
        )

    def forecast(self, hist, horizon):
        close, idx, _ = _prepare(hist, horizon, self.lookback + 1, self.validate)
        n = len(close) - 1
        if n < 1:
            raise ValueError("At least two close prices are needed for returns.")
        # Telescoping sum of log returns: two logs instead of a full series.
        mean_return = (np.log(close[-1]) - np.log(close[0])) / n
        cumulative = np.arange(1, horizon + 1) * mean_return
        return _price_forecast(close[-1], cumulative, idx)


class Arima:
    """ARIMA on daily log returns with speed options.

    Parameters
    ----------
    order : (p, d, q), fixed. Select on pre-test data, then freeze.
    lookback : recent daily returns used for estimation.
    refit_every : re-estimate parameters every N forecast calls.
        Between refits, new returns are filtered through the frozen
        parameters (state stays current, coefficients are stale).
    warm_start : start the optimizer from the previous parameters.
    filter_window : returns filtered between refits (None = lookback).
        Shorter is faster and nearly identical, but not exact.
    scale : returns are multiplied by this before fitting and forecasts are
        divided by it afterwards. 100 improves conditioning. Use 1.0 to
        reproduce unscaled fits.
    concentrate_scale : profile the innovation variance out of the
        likelihood (one fewer parameter). Off by default: validate against
        the reference run before enabling, including with refit_every > 1.
    validate : False checks only the last row (walk_forward already
        validated the full series).

    Estimation never computes parameter standard errors (cov_type="none").
    Fit one reference model with the default covariance on pre-test data if
    you need coefficients and significance for reporting.

    Failure handling: a failed fit triggers one cold refit. If that also
    fails, the forecast is zero (Naive). See config["n_fallbacks"] and
    config["n_zero_forecasts"].

    The instance keeps state between calls. Call reset() between tickers
    (or use one instance per ticker). A call whose history does not move
    forward in time forces a refit and counts as a new run.
    """
    name = "arima"

    def __init__(self, order=(1, 0, 1), lookback=1000, refit_every=1,
                 warm_start=False, filter_window=None, scale=100.0,
                 concentrate_scale=False, validate=True):
        _validate_lookback(lookback)
        _validate_positive_int("refit_every", refit_every)
        if filter_window is not None:
            _validate_positive_int("filter_window", filter_window)
        if len(order) != 3 or any(
            isinstance(v, bool) or not isinstance(v, Integral) or v < 0
            for v in order
        ):
            raise ValueError("order must contain three nonnegative integers.")
        if (isinstance(scale, bool) or not isinstance(scale, Real)
                or not np.isfinite(scale) or scale <= 0):
            raise ValueError("scale must be a positive number.")

        self.order = tuple(int(v) for v in order)
        self.lookback = int(lookback)
        self.refit_every = int(refit_every)
        self.warm_start = bool(warm_start)
        self.filter_window = None if filter_window is None else int(filter_window)
        self.scale = float(scale)
        self.concentrate_scale = bool(concentrate_scale)
        self.validate = bool(validate)
        self.reset()

    @property
    def config(self):
        return dict(
            model=self.name,
            target="log_return",
            order=self.order,
            lookback=self.lookback,
            refit_every=self.refit_every,
            warm_start=self.warm_start,
            filter_window=self.filter_window,
            scale=self.scale,
            concentrate_scale=self.concentrate_scale,
            cov_type="none",
            n_fits=self._n_fits,
            n_fallbacks=self._n_fallbacks,
            n_zero_forecasts=self._n_zero,
        )

    def reset(self):
        """Clear the cached fit and counters (call between tickers)."""
        self._fit = None
        self._params = None
        self._last_ts = None
        self._since_fit = 0
        self._n_fits = 0
        self._n_fallbacks = 0
        self._n_zero = 0

    def _estimate(self, y, cold=False):
        """Estimate parameters on scaled returns y, warm-starting if possible."""
        model = ARIMA(y, order=self.order,
                      concentrate_scale=self.concentrate_scale)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.warm_start and not cold and self._params is not None:
                try:
                    fit = model.fit(start_params=self._params, cov_type="none")
                except Exception:
                    fit = model.fit(cov_type="none")
            else:
                fit = model.fit(cov_type="none")
        self._n_fits += 1
        return fit

    def _forecast_from(self, fit, horizon):
        daily = np.asarray(fit.forecast(steps=horizon), dtype=float) / self.scale
        if not np.isfinite(daily).all():
            raise FloatingPointError("non-finite forecast")
        return daily

    def _daily_forecast(self, r, last_ts, horizon):
        """Daily return forecasts, or None if every attempt fails."""
        y = r * self.scale
        need_fit = (
            self._fit is None
            or self._last_ts is None
            or last_ts <= self._last_ts          # new ticker or restart
            or self._since_fit >= self.refit_every
        )
        try:
            if need_fit:
                self._fit = self._estimate(y)
                self._since_fit = 0
            else:
                window = y if self.filter_window is None else y[-self.filter_window:]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._fit = self._fit.apply(window, refit=False)
            return self._forecast_from(self._fit, horizon)
        except Exception:
            self._n_fallbacks += 1
            try:
                self._fit = self._estimate(y, cold=True)
                self._since_fit = 0
                return self._forecast_from(self._fit, horizon)
            except Exception:
                self._fit = None
                return None

    def forecast(self, hist, horizon):
        close, idx, last_ts = _prepare(hist, horizon, self.lookback + 1,
                                       self.validate)
        r = _returns(close)

        daily = self._daily_forecast(r, last_ts, horizon)
        if daily is None:
            self._n_zero += 1
            daily = np.zeros(horizon)
        else:
            self._params = np.asarray(self._fit.params, dtype=float)

        self._since_fit += 1
        self._last_ts = last_ts
        return _price_forecast(close[-1], np.cumsum(daily), idx)


ALL_BASELINES = [Naive, Drift, Arima]