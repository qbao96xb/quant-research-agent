# Tool Interface

`src/tools/registry.call()` is the single entry point. Every call is validated against a Pydantic schema in `src/tools/schemas.py`.

**The table below is a snapshot and known to be incomplete (the registry has ~15 tools). Always confirm against `registry.py` and `schemas.py`.**

## Example

```python
from src.tools.registry import call

ds = call("load_data", ticker="KO", start="2015-01-01")["dataset_id"]
bt = call("run_backtest", dataset_id=ds, models=["naive", "arima"], h=5,
          start="2025-01-01", step=5)["backtest_id"]
metrics = call("get_metrics", backtest_id=bt, model="arima")
dm = call("dm_test", backtest_id=bt, model_a="arima")
```

## Core Tools

| Tool | Schema | Description | Slow |
|---|---|---|---|
| `load_data` | `LoadDataArgs` | Fetch OHLCV for a ticker | No |
| `describe_data` | `DescribeArgs` | Dataset summary | No |
| `forecast` | `ForecastArgs` | Single-model forecast at one origin | No |
| `forecast_path` | `ForecastPathArgs` | Dated multi-day forecast path | No |
| `garch_volatility` | `GarchVolatilityArgs` | GARCH volatility forecast; `history` = last 60 days of in-sample conditional daily vol | No |
| `run_backtest` | `BacktestArgs` | Walk-forward backtest, may run in background | Yes |
| `backtest_status` | - | Poll a background backtest | No |
| `get_metrics` | `MetricsArgs` | MAE, RMSE, dir_acc, coverage | No |
| `dm_test` | `DMArgs` | Diebold-Mariano test | No |
| `tune_arima` | `TuneArimaArgs` | ARIMA order search (`max_p`, `max_q` 0-3; `n_val`, `n_test`). Returns `best_order`, `table`, `test_rmse`, `dm_best_vs_naive`, `dm_best_vs_default`, `held_out_check`, `check_reason`, `cfg`, `note`, `tune_id`, `attempt` (tuning runs on this data and `h`; above 1 adds a re-tune warning to `note`) | No |
| `tune_kronos` | `TuneKronosArgs` | Kronos config search on validation origins | Yes |
| `list_artifacts` | `ListArgs` | List cached artifacts | No |

## Key Schemas (snapshot)

**LoadDataArgs**: `ticker` (required), `start` (default `2015-01-01`), `end` (optional)

**BacktestArgs**: `dataset_id`, `models` (default `["naive","drift","arima"]`), `h` (1-20, default 5), `start` (date or negative row offset, default `2025-01-01`), `step` (1-60, default 5), `order` (ARIMA only)

**ForecastPathArgs**: singular `model`; `order` only valid with `model="arima"`

**DMArgs**: `backtest_id`, `model_a` (candidate), `model_b` (default `naive`), `h`

## Conventions

- `ModelName` literal (built in `src/tools/model_names.py`): `naive`, `drift`, `arima`, `kronos_mini`, `kronos_small`, `kronos_base`, and `kronos_small_ft` only when `outputs/kronos_ft_small_listc/basemodel/best_model` exists
- `order` is accepted only for ARIMA; other models must reject it
- Tool outputs are the only valid source of numbers in agent answers
- New tools need: schema, registry entry, unit test, and an agent eval case before the prompt mentions them
