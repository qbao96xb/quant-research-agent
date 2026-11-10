# src/tools/schemas.py
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator
from .model_names import MODEL_NAMES, KRONOS_SIZES
ModelName = Literal.__getitem__(MODEL_NAMES)
KronosSize = Literal.__getitem__(KRONOS_SIZES)

class LoadDataArgs(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, for example KO")
    start: str = Field("2015-01-01", description="Start date, YYYY-MM-DD")
    end: Optional[str] = Field(None, description="End date, YYYY-MM-DD, defaults to latest")

class DescribeArgs(BaseModel):
    dataset_id: str = Field(..., description="ID returned by load_data")

def _check_order(order, uses_arima):
    if order is None:
        return
    if not uses_arima:
        raise ValueError("order is only valid for the arima model")
    p, d, q = order
    if d != 0 or not (0 <= p <= 3 and 0 <= q <= 3):
        raise ValueError("order must be (p, 0, q) with p and q from 0 to 3")

class ForecastArgs(BaseModel):
    dataset_id: str = Field(..., description="ID returned by load_data")
    model: ModelName = Field(..., description="Forecast model")
    h: int = Field(5, ge=1, le=20, description="Horizon in trading days")
    level: float = Field(0.90, ge=0.50, le=0.99,
                         description="Interval level, e.g. 0.80, 0.90, 0.95. Default 0.90.")
    origin: Optional[str] = Field(None, description="YYYY-MM-DD, defaults to the last date")
    order: Optional[tuple[int, int, int]] = Field(
        None, description="ARIMA order (p, 0, q). Only with model='arima'. Use best_order from tune_arima.")

    @model_validator(mode="after")
    def _order_ok(self):
        _check_order(self.order, self.model == "arima")
        return self

class ForecastPathArgs(BaseModel):
    dataset_id: str = Field(..., description="ID returned by load_data")
    model: ModelName = Field(..., description="Forecast model")
    h: int = Field(10, ge=1, le=20,
                   description="Number of days in the path (days 1..h) from the origin")
    level: float = Field(0.90, ge=0.50, le=0.99,
                         description="Interval level, e.g. 0.80, 0.90, 0.95. Default 0.90.")
    origin: Optional[str] = Field(None, description="YYYY-MM-DD, defaults to the last date")
    order: Optional[tuple[int, int, int]] = Field(
        None, description="ARIMA order (p, 0, q). Only with model='arima'. Use best_order from tune_arima.")

    @model_validator(mode="after")
    def _order_ok(self):
        _check_order(self.order, self.model == "arima")
        return self


class GarchVolatilityArgs(BaseModel):
    dataset_id: str = Field(..., description="ID returned by load_data")
    h: int = Field(5, ge=1, le=20, description="Horizon in trading days")
    level: float = Field(0.90, ge=0.50, le=0.99,
                         description="Interval level, e.g. 0.80, 0.90, 0.95. Default 0.90.")


class BacktestArgs(BaseModel):
    dataset_id: str = Field(..., description="ID returned by load_data")
    models: list[ModelName] = Field(
        ..., min_length=1,
        description="Models to compare. Must include 'naive' (the DM benchmark). "
                    "Kronos models are slow, so use few origins.")
    h: int = Field(5, ge=1, le=20, description="Horizon in trading days")
    start: int | str = Field(
        "2025-01-01",
        description="Date YYYY-MM-DD, or negative row offset such as -500. "
                    "Use about -100 when Kronos models are included.")
    step: int = Field(5, ge=1, le=60, description="Trading days between origins")
    order: Optional[tuple[int, int, int]] = Field(
        None, description="ARIMA order (p, 0, q). Only when 'arima' is in models. Use best_order from tune_arima.")

    @model_validator(mode="after")
    def _order_ok(self):
        _check_order(self.order, "arima" in self.models)
        return self


class CompareArgs(BaseModel):
    backtest_id: str = Field(..., description="ID returned by run_backtest")
    benchmark: ModelName = Field("naive", description="Benchmark model")
    h: int = Field(5, ge=1, le=20, description="Horizon used in the backtest")

class MetricsArgs(BaseModel):
    backtest_id: str = Field(..., description="ID returned by run_backtest")
    model: ModelName = Field(..., description="Model to score")

class DMArgs(BaseModel):
    backtest_id: str = Field(..., description="ID returned by run_backtest")
    model_a: ModelName = Field(..., description="Model under test")
    model_b: ModelName = Field("naive", description="Benchmark model")
    h: int = Field(5, ge=1, le=20, description="Horizon used in the backtest")

class StatusArgs(BaseModel):
    backtest_id: str = Field(..., description="ID returned by run_backtest")
    wait_s: int = Field(30, ge=0, le=45, description="Seconds to wait. Max 45. Use 30.")

class TuneKronosArgs(BaseModel):
    dataset_id: str = Field(..., description="ID returned by load_data")
    size: KronosSize = Field("small", description="Kronos size")
    h: int = Field(5, ge=1, le=20, description="Horizon in trading days")
    step: int = Field(5, ge=1, le=60, description="Trading days between origins")
    val_origins: int = Field(30, ge=30, le=100, description="Validation origins (min 30)")
    test_origins: int = Field(12, ge=5, le=60, description="Origins held out for the final test")

class TuneArimaArgs(BaseModel):
    dataset_id: str
    h: int = Field(5, ge=1, le=20)
    max_p: int = Field(2, ge=0, le=3)
    max_q: int = Field(2, ge=0, le=3)
    n_val: int = Field(60, ge=20, le=200)
    n_test: int = Field(40, ge=20, le=100)

class ListArgs(BaseModel):
    pass


class ReportArgs(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, for example KO")
    h: int = Field(5, ge=1, le=20, description="Forecast horizon in trading days")
    models: list[ModelName] = Field(
        default_factory=lambda: ["naive", "drift", "arima"],
        description="Models to compare. Must include naive. Avoid kronos models (slow).")
    start: str = Field("2015-01-01", description="Data start, YYYY-MM-DD")


class TrackArgs(BaseModel):
    dataset_id: str = Field(..., description="ID returned by load_data")
    model: ModelName = Field(..., description="Forecast model")
    n: int = Field(30, ge=5, le=120, description="Number of most recent origins to evaluate")
    h: int = Field(1, ge=1, le=20, description="Horizon in trading days for each forecast")
    order: Optional[tuple[int, int, int]] = Field(
        None, description="ARIMA order (p, 0, q). Only with model='arima'.")

    @model_validator(mode="after")
    def _order_ok(self):
        _check_order(self.order, self.model == "arima")
        return self
