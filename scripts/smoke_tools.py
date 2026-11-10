"""Smoke test: call every tool once and print a short result."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.registry import call, REGISTRY


def show(name, r, keys=("ok", "error")):
    short = {k: r.get(k) for k in keys if k in r}
    print(f"{name:<18} {short}")
    return r


print("Registered tools:", sorted(REGISTRY))

# 1. load_data
r = show("load_data", call("load_data", ticker="KO", start="2015-01-01"), ("ok", "dataset_id", "rows"))
assert r["ok"], r
ds = r["dataset_id"]

# 2. describe_data (skip if not in REGISTRY)
if "describe_data" in REGISTRY:
    assert show("describe_data", call("describe_data", dataset_id=ds))["ok"]

# 3. forecast
f = call("forecast", dataset_id=ds, model="arima", h=5)
show("forecast", f)
assert f["ok"], f

# 4. garch_volatility
if "garch_volatility" in REGISTRY:
    g = call("garch_volatility", dataset_id=ds, h=5)
    show("garch_volatility", g)
    assert g["ok"], g

# 5. run_backtest
b = call("run_backtest", dataset_id=ds, start=-500, h=5, step=5)
show("run_backtest", b, ("ok", "backtest_id", "error"))
assert b["ok"], b
bt = b["backtest_id"]

# 6. get_metrics
m = call("get_metrics", backtest_id=bt, model="arima")
show("get_metrics", m)
assert m["ok"], m

# 7. dm_test
d = call("dm_test", backtest_id=bt, model_a="arima", model_b="naive")
show("dm_test", d, ("ok", "n", "dm", "verdict", "error"))
assert d["ok"], d

# 8. list_artifacts
a = call("list_artifacts")
show("list_artifacts", a)
assert a["ok"], a

print("\nAll tool calls returned ok=True")
