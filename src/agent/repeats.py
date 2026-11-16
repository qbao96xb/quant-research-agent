"""Stop on a turn that only repeats calls which already succeeded (S03, owner option A, 2026-11-16).

At temperature 0 the agent model sometimes answers a finished tool call by calling it again
with the same arguments. Such a turn means it has what it needs: the loop skips the turn and
asks for the final answer without tools, instead of sending cached results back.
"""

REPEAT_MSG = ("You already have the results of these calls. Stop calling tools and give "
              "the final answer from the results you already have.")


def only_repeats(calls, cache, key, log, run_id, step):
    """True when the turn has calls and every one has an identical successful call in `cache`."""
    keys = [key((tc.get("function") or {}).get("name") or "unknown",
                (tc.get("function") or {}).get("arguments") or "{}") for tc in calls]
    if not keys or not all(k in cache for k in keys):
        return False
    log(run_id=run_id, step=step, event="dup_escalation", duplicates=len(keys))
    return True
