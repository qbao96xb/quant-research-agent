"""Tool results for the LLM: at most about 8,000 characters and always valid JSON.

A plain character cut breaks the JSON and drops the fields at the end (for example the
`summary` of forecast_vs_actual). Here the longest list is halved until the text fits,
and a "truncated" field says how many items of each cut list are shown.
"""

import json

from .guardrails import truncate

LIMIT = 8000


def fit_json(result, n=LIMIT):
    s = json.dumps(result)
    if len(s) <= n or not isinstance(result, dict):
        return truncate(s, n)
    out = dict(result)
    while len(s) > n:
        lists = [k for k, v in out.items() if isinstance(v, list) and v]
        if not lists:
            return truncate(s, n)
        k = max(lists, key=lambda key: len(json.dumps(out[key])))
        out[k] = out[k][: len(out[k]) // 2]
        out["truncated"] = {key: f"showing the first {len(out[key])} of {len(result[key])}"
                            for key in result
                            if isinstance(result[key], list) and len(out[key]) < len(result[key])}
        s = json.dumps(out)
    return s
