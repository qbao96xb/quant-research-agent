import json
import pytest
from src.tools.registry import REGISTRY, call, schemas

EXPECTED = {
    "backtest_status", "compare_models", "describe_data", "dm_test",
    "forecast", "garch_volatility", "get_metrics", "list_artifacts",
    "load_data", "run_backtest", "tune_kronos", "tune_arima", "forecast_path",
    "make_report", "forecast_vs_actual"
}


def _schema_name(s):
    # handles flat {"name": ...} and OpenAI-style {"function": {"name": ...}}
    return s.get("name") or s.get("function", {}).get("name")


def test_expected_tools_registered():
    assert set(REGISTRY) == EXPECTED


@pytest.mark.parametrize("name", list(REGISTRY))
def test_docstring(name):
    assert REGISTRY[name]["fn"].__doc__, name


def test_schemas_serializable():
    json.dumps(schemas())


def test_schemas_match_registry():
    names = [_schema_name(s) for s in schemas()]
    assert len(names) == len(set(names)), "duplicate schema names"
    assert set(names) == set(REGISTRY)


def test_unknown_tool():
    assert call("nope")["error"]["code"] == "UNKNOWN_TOOL"


def test_bad_args_return_error_not_raise():
    r = call("load_data")  # missing required ticker
    assert r["ok"] is False and "error" in r


def test_list_artifacts():
    r = call("list_artifacts")
    assert r["ok"] and isinstance(r["artifacts"], list)
