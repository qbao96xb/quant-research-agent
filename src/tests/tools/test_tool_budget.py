from types import SimpleNamespace

import pytest

from src.tools.errors import ToolError
from src.tools.registry import call
from src.tools.tool_budget import FIT_BUDGET, check_tune_budget, tune_fits


def args(max_p=2, max_q=2, n_val=60, n_test=40):
    return SimpleNamespace(max_p=max_p, max_q=max_q, n_val=n_val, n_test=n_test)


def test_fit_counts():
    assert tune_fits(2, 2, 60, 40) == 620          # defaults
    assert tune_fits(3, 3, 60, 40) == 1040         # full grid, default splits
    assert tune_fits(3, 3, 200, 100) == 3400       # the T02 request


def test_budget_is_1200():
    assert FIT_BUDGET == 1200


def test_within_budget_passes():
    assert check_tune_budget(args()) == 620
    assert check_tune_budget(args(max_p=3, max_q=3)) == 1040


def test_over_budget_is_refused():
    with pytest.raises(ToolError) as e:
        check_tune_budget(args(max_p=3, max_q=3, n_val=200, n_test=100))
    assert e.value.code == "OVER_BUDGET"
    assert "3400" in e.value.message


def test_tool_refuses_before_loading_data():
    r = call("tune_arima", dataset_id="ds_missing", max_p=3, max_q=3, n_val=200, n_test=100)
    assert r["ok"] is False
    assert r["error"]["code"] == "OVER_BUDGET"
