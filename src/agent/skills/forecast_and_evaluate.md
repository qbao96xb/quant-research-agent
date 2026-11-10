# Skill: forecast and evaluate
Use when: the user asks whether a model is accurate, better than naive, or asks to evaluate or compare. Do NOT use for a plain forecast request.
Not for: risk or volatility-only questions (use the volatility skill).

## Defaults
Use the user's value if given. Otherwise:
- model `arima`, h `5`, data start `2015-01-01`, backtest start -500, step `1`
- Ask only if the ticker is missing.

## Scope
- **Forecast only** ("forecast KO", "forecast KO with ARIMA"): steps 1-2, then answer. Do NOT run steps 3-4.
- **Accuracy or comparison** ("compare to naive", "is it any good"): steps 1-4.

## Steps
1. `load_data(ticker, start)` -> `dataset_id`. Skip if you already have one for this ticker.
2. `forecast(dataset_id, model, h)` -> mean, lo, hi.
3. `run_backtest(dataset_id, start, h, step)` -> `backtest_id`. Slow, so run it once and reuse the ID.
4. `compare_models(backtest_id)` -> `table` with metrics and DM p-value for all models.
5. Answer in the standard format: forecast, interval, the pasted `table`, then verdict and caveats.

## Gotchas
- **A forecast alone does not need a backtest.** Run steps 3-4 only when the user asks about accuracy, quality, or comparison with naive.
- **Lower MAE is not a win.** Only the DM p-value in the table decides. Above 0.05 means "no significant difference".
- **`NOT_ENOUGH_DATA` from the backtest:** reload with an earlier data start, then retry once.
- **Cov90 near 0.90** means the interval is calibrated. Report Width too, since wide intervals cover easily.

## Example
User: "Forecast KO for 5 days and compare to naive"
1. `load_data(ticker="KO", start="2015-01-01")` -> `ds_a1`
2. `forecast(dataset_id="ds_a1", model="arima", h=5)`
3. `run_backtest(dataset_id="ds_a1", start=-500, h=5, step=5)` -> `bt_b2`
4. `compare_models(backtest_id="bt_b2")` -> `table`
