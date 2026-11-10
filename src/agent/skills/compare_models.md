# Skill: compare models
Use when: the user asks which model is better, or to rank or compare models.
Not for: a single forecast (use the forecast skill).
Scope: one ticker, time series, point forecasts of h-day log returns.

## Defaults
Use the user's values if given. Otherwise:
- models `drift`, `arima`, `kronos`, plus `naive` as the benchmark
- h `5`, data start `2015-01-01`, backtest start -500, step `1`
- Ask only if the ticker is missing.

## Steps
1. `load_data(ticker, start)` -> `dataset_id`. Skip if you already have one for this ticker.
2. `run_backtest(dataset_id, start, h, step)` -> `backtest_id`. Run it once and reuse the ID.
3. `compare_models(backtest_id)` once. It returns a `table` with metrics and DM p-value for every model.
4. Paste the `table` verbatim, then the verdict, then the caveats.
5. Present the table, then the verdict, then the caveats.

## Table
- Paste the `table` field from `compare_models` verbatim. Do not retype or re-round the numbers.
- The `naive` row shows "-" for DM.
- Add `n` once under the table, from the `n` field.

## Verdict rules
Read each model's DM p-value in the table against naive:
- **p > 0.05:** "no significant difference from naive."
- **p <= 0.05 and DM < 0:** significantly better than naive.
- **p <= 0.05 and DM > 0:** significantly worse than naive.
- **No model is significant:** say "no model beats naive". Do not rank by MAE.
- **Two non-naive models:** only call `dm_test` between them if the user asks for that pair.

## Interpretation
- **Returns are noisy.** Differences between models are usually tiny. Report the gap and the p-value, not "X is X% better".
- **Judge by DM, not by one metric.** MAE, RMSE and DirAcc can disagree. The DM test (squared-error loss) is the deciding evidence.
- **DirAcc is easy to inflate.** In a rising market, always predicting "up" scores above 0.50. Treat DirAcc as supporting evidence only.
- **Significant is not profitable.** The tools measure forecast accuracy, not trading results. Always say so when a model is significantly better.

## Gotchas
- **DM sign:** only needed when p <= 0.05. Call `dm_test` for that model. A positive DM means `model_a` has the larger error.
- **Several tests:** with 3 models, one p just under 0.05 can be chance. Say so when it is the only significant result.
- **Cov90 and Width go together:** a wide interval covers easily. Do not praise Cov90 alone.
- **Overlapping targets:** when `step` < `h`, the observations overlap. Mention it in the caveats.
- **`UNKNOWN_ID` or a missing model:** run `list_artifacts`, then retry once. If `kronos` is missing from the backtest, say so and drop it from the table.

## Not available (say so, never estimate)
- R², MAPE, rank correlation (IC), Sharpe, Sortino, max drawdown, transaction costs, portfolio or long/short results.
- If asked: "These tools measure forecast accuracy only. I can't report <metric>."
- Cross-sectional ranking of several tickers is not supported. Offer one backtest per ticker instead.

## Example
User: "Which model is best for KO?"
1. `load_data(ticker="KO", start="2015-01-01")` -> `ds_a1`
2. `run_backtest(dataset_id="ds_a1", start=-500, h=5, step=5)` -> `bt_b2`
3. `compare_models(backtest_id="bt_b2")` -> `table`
4. Answer: the table, then "no significant difference from naive for any model", then caveats (n, accuracy is not trading profit, research only).
