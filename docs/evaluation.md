# Evaluation

Two separate systems. Do not mix their holdouts.

## 1. Forecast Evaluation (`src/evaluation/`)

### Walk-forward backtest (`walk_forward.py`)
- Rolling origins from `start`, spaced by `step`, horizon `h`
- History ends at or before the origin; targets strictly after it (no look-ahead)
- Needs `min_hist` = 400 rows before the first origin
- Output long DataFrame: `origin, h, target_date, last, actual, pred, lower, upper`

### Metrics (`metrics.py`)
- All on cumulative log returns `log(pred/last)` from the origin
- MAE, RMSE, directional accuracy (`dir_acc`), interval coverage, interval width
- Out-of-sample RMSE vs naive is the primary selection criterion; AIC/BIC are in-sample screening only

### Diebold-Mariano (`dm_test.py`)
- Squared or absolute loss, Newey-West HAC, Harvey-Leybourne-Newbold correction
- Needs at least 10 paired observations
- `step < h` creates overlapping targets; HAC is required
- Default HAC lags are `ceil(h / step) - 1`; `run_backtest` uses `step = h`, so lags are 0 there
- Clark-West (`cw_test.py`): MSPE-adjusted, one-sided (H1: candidate better), Newey-West variance with the same lags, standard normal p-value; `cw_table` runs it only for models that nest naive (`drift`, `arima`); `compare_models` shows it as "CW p-value"
- Drift reference: when `drift` is in the run, `compare_models` adds "DM p vs drift" and reports add `dm_drift` / `verdict_drift`; naive stays the primary benchmark

### Model tuning
- ARIMA: grid over `p,q`, fixed `d=0`, select on validation RMSE, tie-break smaller `p+q`, evaluate once on test
- Never tune on the test split; do not change `ARIMA_CFG` or `SEARCH_CFG` defaults without rerunning comparisons

## 2. Agent Evaluation (`src/agent_eval/`)

- `cases.yaml`: dev and test cases, frozen; new cases go in `cases_v2.yaml`
- Scorer decides pass/fail; neither model's opinion counts
- Tracked: pass rate by category, hallucination/grounding rate, latency, steps

### Commands
```bash
PYTHONPATH=src:. python -m scripts.eval_agent --name NAME --model MODEL --only S01 --n 1   # one case
PYTHONPATH=src:. python -m scripts.eval_agent --name NAME --model MODEL --n 3              # full dev split
PYTHONPATH=src:. python -m src.agent_eval.compare --n 3 --split dev                        # all models in MODELS
```
`--split test` is the held-out split: owner approval required.

### Rules
- Iterate on the dev split only; the owner runs test once at the end
- Freeze prompt and scorer before any held-out run
- Compare only runs with the same model, prompt version, scorer and cases; re-score old runs with current cases first
- Accept a change only if the dev pass rate rises and the hallucination rate does not
- Each merged branch should have a before/after eval comparison attached
