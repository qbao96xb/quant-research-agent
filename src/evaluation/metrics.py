"""Accuracy metrics for cumulative log-return forecasts.

Price columns remain the source of truth:
    pred, actual, last, lower, upper

Returns are cumulative from the forecast origin, in decimal units.
"""
import numpy as np
import pandas as pd


def add_returns(res: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative log returns and forecast errors from prices."""
    d = res.copy()

    required = {"pred", "actual", "last"}
    missing = required.difference(d.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for column in required:
        values = d[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{column} must contain finite, positive prices.")

    d["pred_return"] = np.log(d["pred"] / d["last"])
    d["actual_return"] = np.log(d["actual"] / d["last"])

    for price_column, return_column in [
        ("lower", "lower_return"),
        ("upper", "upper_return"),
    ]:
        if price_column not in d:
            d[price_column] = np.nan

        values = d[price_column].to_numpy(dtype=float)
        present = ~np.isnan(values)
        if (
            not np.isfinite(values[present]).all()
            or (values[present] <= 0).any()
        ):
            raise ValueError(
                f"{price_column} must contain positive finite prices or NaN."
            )

        d[return_column] = np.log(d[price_column] / d["last"])

    both = d["lower_return"].notna() & d["upper_return"].notna()
    if (
        d.loc[both, "lower_return"] > d.loc[both, "upper_return"]
    ).any():
        raise ValueError("Lower bounds cannot exceed upper bounds.")

    # Backward-compatible aliases.
    d["r_pred"] = d["pred_return"]
    d["r_act"] = d["actual_return"]
    d["err"] = d["pred_return"] - d["actual_return"]
    return d


def score(
    res: pd.DataFrame,
    by=("ticker", "model", "h"),
) -> pd.DataFrame:
    """Return accuracy and interval metrics, grouped by available keys.

    width is cumulative log-return interval width.
    Zero predicted or realized returns are excluded from directional accuracy.
    """
    d = add_returns(res)
    group_columns = [column for column in by if column in d.columns]

    direction_valid = (
        d["pred_return"].ne(0)
        & d["actual_return"].ne(0)
    )
    d["dir_ok"] = np.nan
    d.loc[direction_valid, "dir_ok"] = (
        np.sign(d.loc[direction_valid, "pred_return"])
        == np.sign(d.loc[direction_valid, "actual_return"])
    ).astype(float)

    interval_valid = (
        d["lower_return"].notna()
        & d["upper_return"].notna()
    )
    d["covered"] = np.nan
    d.loc[interval_valid, "covered"] = (
        (
            d.loc[interval_valid, "actual_return"]
            >= d.loc[interval_valid, "lower_return"]
        )
        & (
            d.loc[interval_valid, "actual_return"]
            <= d.loc[interval_valid, "upper_return"]
        )
    ).astype(float)

    d["width"] = d["upper_return"] - d["lower_return"]
    # Width requires both endpoints, just like coverage.
    d.loc[~interval_valid, "width"] = np.nan

    # Allow an ungrouped score when none of the requested keys exist.
    if not group_columns:
        d["_group"] = "all"
        group_columns = ["_group"]

    return d.groupby(group_columns, dropna=False).agg(
        n=("err", "size"),
        MAE=("err", lambda e: float(e.abs().mean())),
        RMSE=("err", lambda e: float(np.sqrt((e ** 2).mean()))),
        dir_acc=("dir_ok", "mean"),
        n_direction=("dir_ok", "count"),
        coverage=("covered", "mean"),
        n_intervals=("covered", "count"),
        width=("width", "mean"),
    )


def errors(res: pd.DataFrame, h: int, ticker=None) -> pd.Series:
    """Return sorted errors for one ticker/model/horizon, indexed by origin.

    Duplicate origins are rejected to prevent ambiguous DM pairing.
    """
    d = add_returns(res)
    d = d.loc[d["h"] == h]

    if ticker is not None:
        if "ticker" not in d:
            raise ValueError("ticker filtering requires a ticker column.")
        d = d.loc[d["ticker"] == ticker]

    if d["origin"].isna().any():
        raise ValueError("Forecast origins cannot be missing.")

    if d["origin"].duplicated().any():
        raise ValueError(
            "Duplicate origins: select one ticker and one model before "
            "extracting errors."
        )

    return d.set_index("origin")["err"].sort_index()
