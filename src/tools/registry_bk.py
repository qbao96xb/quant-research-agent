# src/tools/registry.py
from pydantic import ValidationError

from .errors import ToolError
from .schemas import (LoadDataArgs, DescribeArgs, ForecastArgs, ForecastPathArgs,
                      GarchVolatilityArgs, BacktestArgs, MetricsArgs, DMArgs, ListArgs,
                      StatusArgs, CompareArgs, TuneKronosArgs)
from .data_tools import load_data, describe_data
from .forecast_tools import forecast, forecast_path, garch_volatility
from .eval_tools import run_backtest, get_metrics, dm_test, compare_models, backtest_status_tool
from .tune_kronos_tools import tune_kronos
from .cache import list_artifacts

REGISTRY = {
    "load_data":       dict(fn=load_data,       args=LoadDataArgs, slow=False),
    "describe_data":   dict(fn=describe_data,   args=DescribeArgs, slow=False),
    "forecast":        dict(fn=forecast,        args=ForecastArgs, slow=False),
    "forecast_path":   dict(fn=forecast_path,   args=ForecastPathArgs, slow=False),
    "garch_volatility": dict(fn=garch_volatility, args=GarchVolatilityArgs, slow=False),
    "run_backtest":    dict(fn=run_backtest,    args=BacktestArgs, slow=True),
    "tune_kronos": dict(fn=tune_kronos, args=TuneKronosArgs, slow=True),
    "get_metrics":     dict(fn=get_metrics,     args=MetricsArgs,  slow=False),
    "compare_models": dict(fn=compare_models, args=CompareArgs, slow=False),
    "backtest_status": dict(fn=backtest_status_tool,  args=StatusArgs,  slow=True),
    "dm_test":         dict(fn=dm_test,         args=DMArgs,       slow=False),
    "list_artifacts":  dict(fn=list_artifacts,  args=ListArgs,     slow=False),
}

def call(name: str, **kwargs) -> dict:
    spec = REGISTRY.get(name)
    if spec is None:
        return ToolError("UNKNOWN_TOOL", f"No tool named {name!r}",
                         f"Options: {sorted(REGISTRY)}").to_dict()
    try:
        return spec["fn"](spec["args"](**kwargs))
    except ToolError as e:
        return e.to_dict()
    except ValidationError as e:
        return ToolError("BAD_ARGUMENT", str(e.errors()[0]["msg"]),
                         f"Field: {e.errors()[0]['loc']}").to_dict()
    except Exception as e:
        return ToolError("INTERNAL", f"{type(e).__name__}: {e}").to_dict()

def schemas() -> list[dict]:
    return [dict(name=n, description=s["fn"].__doc__, parameters=s["args"].model_json_schema())
            for n, s in REGISTRY.items()]
