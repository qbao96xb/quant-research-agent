# Skill: volatility analysis
Use when: the user asks about risk, volatility, or the width of the forecast interval.
Not for: return forecasts or model comparisons (use the forecast or compare skills).

## Defaults
Use the user's value if given. Otherwise:
- h `5`, data start `2015-01-01`
- Ask only if the ticker is missing.

## Steps
1. `load_data(ticker, start)` -> `dataset_id`. Skip if you already have one for this ticker.
2. `garch_volatility(dataset_id, h)` -> sigma over the horizon and 90% bounds.
3. Answer in the standard format: sigma, 90% bounds, model name.

## Gotchas
- **Sigma is for the whole h-day return,** not one day. Say "over 5 days", not "daily".
- **Do not annualize or rescale yourself.** If the user wants another unit, say the tool does not provide it.
- **Volatility is not direction.** Never turn the bounds into a price call such as "KO will rise".
- **Name the model once:** GARCH(1,1) with t-distributed errors, fitted on the recent history. Its estimate changes with the data window, so report the data start.
- **Not available as a tool:** accuracy of the volatility forecast against realized volatility. If asked, say so. Do not estimate it.
- **`NOT_ENOUGH_DATA`:** reload with an earlier start, then retry once.

## Example
User: "How risky is KO over the next 5 days?"
1. `load_data(ticker="KO", start="2015-01-01")` -> `ds_a1`
2. `garch_volatility(dataset_id="ds_a1", h=5)`
3. Answer: sigma over 5 days, 90% bounds (`garch_volatility`), GARCH(1,1) with t errors, research only.
