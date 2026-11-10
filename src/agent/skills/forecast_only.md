# Skill: forecast only
Use when: the user asks for a forecast and does not mention accuracy, evaluation, comparison, or naive.
1. `load_data(ticker)`
2. `forecast(dataset_id, model, h)` -> mean, lo, hi.
3. Answer with the forecast and 90% interval. Add the research-only note.
Do not call `run_backtest`, `compare_models`, `get_metrics` or `dm_test`.
