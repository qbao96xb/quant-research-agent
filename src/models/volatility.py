"""Conditional volatility models and interval construction.

Purpose
-------
Fit a GARCH-family model on returns up to a forecast origin, turn the variance
forecast into a cumulative sigma for the h-day return, and use it to build
price intervals or to evaluate volatility forecasts.

Conventions
-----------
- Returns are daily log returns, scaled to percent for fitting (stability).
  All public outputs are rescaled to decimal units (0.01 = 1%).
- Only data up to the origin is used, so there is no look-ahead.
- Sigma is cumulative: element k is the sigma of the k-day return.

Typical use
-----------
>>> vm = VolModel("GARCH", p=1, o=1, q=1, dist="t")   # GJR-GARCH-t
>>> fc = add_intervals(fc, hist, horizon=5, vol_model=vm)
"""
import numpy as np
import pandas as pd
from arch import arch_model

from models.base import Forecast

Z90 = 1.645  # two-sided 90% normal quantile

_VALID_VOL = {"GARCH", "ARCH", "EGARCH"}
_VALID_DIST = {"normal", "t", "skewt", "ged"}
_VALID_MEAN = {"Zero", "Constant", "AR"}


class VolModel:
    """Configurable conditional-volatility model (wrapper around `arch`).

    Parameters
    ----------
    vol : {"GARCH", "ARCH", "EGARCH"}, default "GARCH"
        Variance equation family.
        - "GARCH": standard GARCH. Set ``o>=1`` for GJR-GARCH (leverage term).
        - "ARCH": pure ARCH(p). ``q`` is ignored.
        - "EGARCH": log-variance model. ``o>=1`` gives asymmetric response.
    p : int, default 1
        Lag order of the symmetric innovation term (alpha). Range: ``p >= 1``
        (``p >= 0`` is allowed by `arch` but not meaningful here).
    o : int, default 0
        Lag order of the asymmetric term. Range: ``o >= 0``.
        ``o=0`` is symmetric, ``o=1`` captures the leverage effect
        (negative shocks raise volatility more).
    q : int, default 1
        Lag order of the lagged-variance term (beta). Range: ``q >= 0``.
    dist : {"normal", "t", "skewt", "ged"}, default "t"
        Distribution of the standardized residuals.
        - "normal": thin tails, usually understates tail risk.
        - "t": fat tails, a good default for equities.
        - "skewt": fat tails plus skewness.
        - "ged": generalized error, tail thickness is estimated.
    mean : {"Zero", "Constant", "AR"}, default "Zero"
        Mean equation of daily returns.
        "Zero" is the usual choice at daily frequency, since the mean is
        tiny relative to volatility.
    lookback : int, default 1000
        Number of most recent daily returns used to fit. Range: ``>= 250``
        recommended (about 1 trading year minimum for stable estimates).

    Attributes
    ----------
    name : str
        Label such as ``"GARCH(1,1,1)-t"``, useful in result tables.
    config : dict
        All parameters, ready to save to ``config.json``.

    Raises
    ------
    ValueError
        If ``vol``, ``dist`` or ``mean`` is not a supported value.

    Notes
    -----
    - Choose the specification on pre-test data only (AIC/BIC or QLIKE on a
      validation window), then freeze it for the test window.
    - EGARCH has no analytic multi-step forecast, so ``horizon > 1`` uses
      simulation. Results then vary slightly between runs.

    Examples
    --------
    >>> VolModel()                                   # GARCH(1,1)-t
    >>> VolModel("GARCH", p=2, q=1, dist="normal")   # GARCH(2,1)-N
    >>> VolModel("GARCH", p=1, o=1, q=1, dist="skewt")  # GJR-GARCH-skewt
    >>> VolModel("EGARCH", p=1, o=1, q=1, dist="t")  # EGARCH(1,1,1)-t
    """

    def __init__(self, vol="GARCH", p=1, o=0, q=1, dist="t",
                 mean="Zero", lookback=1000):
        if vol.upper() not in _VALID_VOL:
            raise ValueError(
                f"vol must be one of {sorted(_VALID_VOL)}; "
                "for GJR use vol='GARCH' with o=1.")
        if dist not in _VALID_DIST:
            raise ValueError(f"dist must be one of {sorted(_VALID_DIST)}")
        if mean not in _VALID_MEAN:
            raise ValueError(f"mean must be one of {sorted(_VALID_MEAN)}")
        if p < 0 or o < 0 or q < 0 or lookback < 50:
            raise ValueError("Orders must be >= 0 and lookback >= 50.")

        self.vol, self.p, self.o, self.q = vol.upper(), p, o, q
        self.dist, self.mean, self.lookback = dist, mean, lookback
        self.name = f"{self.vol}({p},{o},{q})-{dist}"
        self.config = dict(vol=self.vol, p=p, o=o, q=q, dist=dist,
                           mean=mean, lookback=lookback)

    def sigma(self, hist, horizon):
        """Forecast cumulative sigma of the k-day log return, k = 1..horizon.

        Parameters
        ----------
        hist : pandas.DataFrame
            Price history up to and including the forecast origin.
            Must contain a ``"close"`` column (adjusted close).
        horizon : int
            Number of days ahead. Range: ``>= 1``. Typical: 1 to 20.

        Returns
        -------
        numpy.ndarray, shape (horizon,)
            Element k-1 is $$\\sqrt{\\sum_{j=1}^{k}\\hat\\sigma^2_{t+j}}$$
            in decimal log-return units (0.02 = 2%).

        Examples
        --------
        >>> VolModel().sigma(hist, 5)
        array([0.011, 0.016, 0.019, 0.022, 0.025])
        """
        r = 100 * np.log(hist["close"]).diff().dropna().iloc[-self.lookback:]
        am = arch_model(r, mean=self.mean, vol=self.vol,
                        p=self.p, o=self.o, q=self.q, dist=self.dist)
        res = am.fit(disp="off")
        if self.vol == "EGARCH" and horizon > 1:
            f = res.forecast(horizon=horizon, method="simulation", reindex=False)
        else:
            f = res.forecast(horizon=horizon, reindex=False)
        var = f.variance.iloc[-1].to_numpy()          # per-step variance, %^2
        return np.sqrt(np.cumsum(var)) / 100


    def diagnostics(self, hist, last=500):
        """Fit on hist and return in-sample series for diagnostic charts.

        Returns a JSON-safe dict (decimal units): dates, ret, cond_vol (daily
        sigma), std_resid, plus params and the dist's nu (None if not fitted).
        Only the last `last` observations are returned to keep the payload small.
        """
        r = 100 * np.log(hist["close"]).diff().dropna().iloc[-self.lookback:]
        res = (arch_model(r, mean=self.mean, vol=self.vol, p=self.p, o=self.o,
                          q=self.q, dist=self.dist).fit(disp="off"))
        tail = slice(-last, None)
        d = hist["timestamps"].iloc[-len(r):].iloc[tail]
        params = {k: float(v) for k, v in res.params.items()}
        return dict(
            model=self.name,
            dates=[str(x)[:10] for x in d],
            ret=(r.iloc[tail] / 100).round(6).tolist(),
            cond_vol=(res.conditional_volatility.iloc[tail] / 100).round(6).tolist(),
            std_resid=res.std_resid.iloc[tail].round(4).tolist(),
            params=params,
            nu=params.get("nu"),
            vol=self.vol, o=self.o,
        )


DEFAULT_VOL = VolModel()


def garch_sigma(hist, horizon, lookback=1000):
    """Cumulative sigma from the default GARCH(1,1)-t (backward-compatible).

    Parameters
    ----------
    hist : pandas.DataFrame
        Price history up to the origin, with a ``"close"`` column.
    horizon : int
        Days ahead, ``>= 1``.
    lookback : int, default 1000
        Returns used for fitting, ``>= 250`` recommended.

    Returns
    -------
    numpy.ndarray, shape (horizon,)
        Cumulative sigma in decimal log-return units.
    """
    return VolModel(lookback=lookback).sigma(hist, horizon)


def add_intervals(fc, hist, horizon, z=Z90, vol_model=DEFAULT_VOL, force=False):
    """Attach volatility-based price intervals to a forecast.

    Interval formula, for forecast day k:
    $$[\\hat p_k\\,e^{-z\\sigma_{1:k}},\\ \\hat p_k\\,e^{+z\\sigma_{1:k}}]$$

    Parameters
    ----------
    fc : models.base.Forecast
        Forecast with ``mean``, ``lower``, ``upper`` (price Series).
    hist : pandas.DataFrame
        Price history up to the origin, used to fit the volatility model.
    horizon : int
        Days ahead, must match the length of ``fc.mean``.
    z : float, default 1.645
        Normal quantile. 1.645 gives a 90% band, 1.96 gives 95%,
        1.28 gives 80%. Range: ``z > 0``.
    vol_model : VolModel, default GARCH(1,1)-t
        Volatility model used to compute sigma.
    force : bool, default False
        - False: only fill bounds that are missing. A model that already has
          both bounds (for example ARIMA) is returned unchanged.
        - True: overwrite bounds for every model, for like-for-like
          coverage comparisons.

    Returns
    -------
    models.base.Forecast
        Same index and mean, with lower and upper bounds filled.

    Examples
    --------
    >>> fc = Naive().forecast(hist, 5)
    >>> fc = add_intervals(fc, hist, 5)                              # default
    >>> fc = add_intervals(fc, hist, 5, vol_model=VolModel("EGARCH"))
    >>> fc = add_intervals(fc, hist, 5, z=1.96, force=True)          # 95%, all models
    """
    both_present = fc.lower.notna().all() and fc.upper.notna().all()
    if both_present and not force:
        return fc
    sig = vol_model.sigma(hist, horizon)
    return Forecast(index=fc.index, mean=fc.mean,
                    lower=fc.mean * np.exp(-z * sig),
                    upper=fc.mean * np.exp(z * sig))


def realized_vol(hist_after, horizon):
    """Realized volatility over the next ``horizon`` days (evaluation target).

    $$RV=\\sqrt{\\sum_{j=1}^{h} r_j^2}$$

    Parameters
    ----------
    hist_after : pandas.DataFrame
        Realized prices starting AT the origin row, then the next days.
        Needs at least ``horizon + 1`` rows and a ``"close"`` column.
        The first row is the origin, so the first return is origin to day 1.
    horizon : int
        Number of days summed, ``>= 1``.

    Returns
    -------
    float
        Realized sigma of the h-day return, decimal units.

    Notes
    -----
    With h = 5 this is a noisy proxy. Use QLIKE, which tolerates this noise.

    Examples
    --------
    >>> rv = realized_vol(df.iloc[t : t + 6], horizon=5)
    """
    r = np.log(hist_after["close"]).diff().dropna().iloc[:horizon]
    return float(np.sqrt((r ** 2).sum()))


def qlike(sigma_hat, sigma_real):
    """QLIKE loss for volatility forecasts (lower is better, minimum 0).

    $$\\text{QLIKE}=\\frac{1}{T}\\sum\\left(\\frac{RV^2}{\\hat\\sigma^2}
    -\\ln\\frac{RV^2}{\\hat\\sigma^2}-1\\right)$$

    Parameters
    ----------
    sigma_hat : array-like of float
        Forecast sigmas, all ``> 0``.
    sigma_real : array-like of float
        Realized sigmas, all ``> 0`` (a zero causes ``log(0)``).
        Same length as ``sigma_hat``.

    Returns
    -------
    float
        Mean QLIKE loss. Penalizes under-forecasting more than
        over-forecasting.

    Examples
    --------
    >>> qlike([0.02, 0.025], [0.021, 0.03])
    0.012
    """
    f = np.asarray(sigma_hat) ** 2
    r = np.asarray(sigma_real) ** 2
    return float(np.mean(r / f - np.log(r / f) - 1))

def evaluate_volatility(
    df,
    specs=None,
    horizon=5,
    start=-500,
    step=5,
    min_hist=400,
    ticker=None,
    output_path=None,
):
    """Evaluate cumulative volatility forecasts against realized volatility.

    start:
        Row position of the first forecast origin.
        Negative values count backward from the end.

    Only history through each origin is used for fitting.
    Future prices are used exclusively to calculate evaluation targets.
    """
    if horizon < 1 or step < 1 or min_hist < 2:
        raise ValueError("Invalid horizon, step, or min_hist.")

    prices = df["close"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Close prices must be finite and positive.")

    specs = [VolModel()] if specs is None else list(specs)

    first = start if start >= 0 else len(df) + start
    first = max(first, min_hist - 1, 0)
    origins = list(range(first, len(df) - horizon, step))

    if not origins:
        raise ValueError("Insufficient data for the requested evaluation.")

    # Include the origin price so the first future return is calculated.
    realized = np.asarray([
        realized_vol(df.iloc[t:t + horizon + 1], horizon)
        for t in origins
    ])

    # This implementation of QLIKE requires strictly positive targets.
    if not np.isfinite(realized).all() or (realized <= 0).any():
        raise ValueError("QLIKE requires finite, positive realized volatility.")

    rows = []
    for vm in specs:
        predicted = np.asarray([
            vm.sigma(df.iloc[:t + 1], horizon)[-1]
            for t in origins
        ])

        if not np.isfinite(predicted).all() or (predicted <= 0).any():
            raise ValueError(f"Invalid volatility predictions from {vm.name}.")

        rows.append({
            "ticker": ticker,
            "vol_model": vm.name,
            "lookback": vm.lookback,
            "horizon": horizon,
            "step": step,
            "n": len(origins),
            "first_origin": df["timestamps"].iloc[origins[0]],
            "last_origin": df["timestamps"].iloc[origins[-1]],
            "qlike": qlike(predicted, realized),
            "sigma_rmse": float(
                np.sqrt(np.mean((predicted - realized) ** 2))
            ),
            "sigma_mae": float(np.mean(np.abs(predicted - realized))),
            "mean_sigma_pred": float(predicted.mean()),
            "mean_sigma_real": float(realized.mean()),
        })

    report = pd.DataFrame(rows)

    if output_path is not None:
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(path, index=False)

    return report

class CachedVol(VolModel):
    """VolModel that fits once per (origin, horizon) and shares it across models.

    Call reset() between tickers.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._cache = {}

    def reset(self):
        self._cache.clear()

    def sigma(self, hist, horizon):
        key = (len(hist), hist["timestamps"].iloc[-1],
               float(hist["close"].iloc[-1]), horizon)
        if key not in self._cache:
            self._cache[key] = super().sigma(hist, horizon)
        return self._cache[key]



if __name__ == "__main__":
    # Usage example: compare candidate specs by QLIKE on the last 250 origins.
    from data.loader import fetch_ohlcv

    H = 5
    df = fetch_ohlcv("KO")
    specs = [
        VolModel("GARCH", 1, 0, 1, "normal"),
        VolModel("GARCH", 1, 0, 1, "t"),
        VolModel("GARCH", 1, 1, 1, "t"),      # GJR-GARCH-t
        VolModel("EGARCH", 1, 1, 1, "t"),
    ]
    origins = range(len(df) - 250, len(df) - H, 5)

    for vm in specs:
        s_hat, s_real = [], []
        for t in origins:
            s_hat.append(vm.sigma(df.iloc[: t + 1], H)[-1])         # h-day sigma
            s_real.append(realized_vol(df.iloc[t : t + H + 1], H))
        print(f"{vm.name:<22} QLIKE = {qlike(s_hat, s_real):.4f}")
