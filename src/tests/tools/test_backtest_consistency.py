import pandas as pd, pytest
from tools.eval_tools import _check_consistent, backtest_key
from tools.errors import ToolError

def _frame(last):
    return pd.DataFrame({"origin": pd.to_datetime(["2025-01-01", "2025-01-08"]),
                         "last": last, "actual": [0.01, 0.02]})

def test_guard_passes_on_identical():
    _check_consistent([_frame([1.0, 2.0]), _frame([1.0, 2.0])])

def test_guard_raises_on_1e7_difference():
    with pytest.raises(ToolError):
        _check_consistent([_frame([1.0, 2.0]), _frame([1.0, 2.0 + 1e-7])])

def test_backtest_key_changes_with_fp():
    a = backtest_key("ds_x", ["naive"], 5, -500, 5, "aaa")
    b = backtest_key("ds_x", ["naive"], 5, -500, 5, "bbb")
    assert a != b
