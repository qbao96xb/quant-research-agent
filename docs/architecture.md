# Architecture

## Data Layer (`src/data/`)

- `loader.py`: fetches OHLCV from Yahoo Finance with local caching
  - Raw parquet in `data/raw/{ticker}.parquet` (latest download)
  - Immutable snapshots in `data/raw/snapshots/{ticker}_{YYYY-MM-DD}.parquet`
  - `manifest.csv` tracks downloads (generated, do not commit)

## Models Layer (`src/models/`)

- `base.py`: `Forecast` dataclass (`mean`, `lower`, `upper`) and `Forecaster` ABC
- `baselines.py`: return-based benchmarks
  - `Naive`: random walk, zero log-return forecast
  - `Drift`: mean of recent log returns
  - `Arima`: ARIMA on daily log returns with refit schedule; default order (1,0,1), tunable
- `volatility.py`: GARCH family via `arch`
  - `VolModel`: GARCH, ARCH, EGARCH; distributions normal, t, skewt, ged
  - `add_intervals()`: volatility-based prediction intervals
- `kronos_model.py`: `KronosForecaster`, `get_forecaster()`, `kronos_forecast()`; sizes mini, small, base, plus `small_ft` when the fine-tuned checkpoint exists (see `src/tools/model_names.py`)
- Fine-tuning: `src/finetune/` (`train_sequential.py`, `finetune_base_model.py`, `finetune_tokenizer.py`, `config_loader.py`); Kronos tuning tool in `src/tools/tune_kronos_tools.py`

## Evaluation Layer (`src/evaluation/`)

- `walk_forward.py`: rolling-origin backtest, see `docs/evaluation.md`
- `metrics.py`: MAE, RMSE, directional accuracy, coverage, interval width
- `dm_test.py`: Diebold-Mariano with Newey-West HAC and HLN correction
- `tuning.py`: ARIMA order search on validation split (`make_splits`, `tune_arima`; tool wrapper in `src/tools/tune_arima_tools.py`)

## Tools Layer (`src/tools/`)

- `schemas.py`: Pydantic models for every tool; source of truth for arguments
- `registry.py`: `call(name, **kwargs)` dispatcher with validation
- `data_tools.py`, `forecast_tools.py`, `eval_tools.py`, `tune_*_tools.py`
- `cache.py`: JSON artifact storage

## Agent Layer (`src/agent/`)

- Agent loop with history trimming, tool timeouts, grounding checks and forced final summary
- LLM client to local Ollama models; model, context length and timeout configured here
- Prompts and skills under `src/agent/prompts`, `src/agent/skills`

## Report Layer

- `src/report/`: report pipeline (`pipeline.py`, `builder.py`, `facts.py`, `tables.py`, `narrative.py`, `number_check.py`)
- `src/report/charts.py`: static Matplotlib charts (Agg backend, saved as PNG at 150 dpi)
- Entry points: `src/tools/report_tools.py` (agent tool), `scripts/make_report.py`

## Data Cache

```
data/
├── raw/
│   ├── {ticker}.parquet
│   └── snapshots/{ticker}_{YYYY-MM-DD}.parquet
├── processed/
└── universe/features.csv
```

## Scripts (`scripts/`)

- `run_agent.py`: agent CLI
- `eval_agent.py`: agent evals (single model, scored; multi-model comparison is `src/agent_eval/compare.py`)
- `build_finetune_data.py`, `run_kronos_smoke.py`: Kronos
- `build_universe.py`, `evaluate.py`, `write_findings.py`
