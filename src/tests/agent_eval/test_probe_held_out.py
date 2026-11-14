from scripts.probe_held_out import END, line, parse

D = {"ok": True, "rows": 2979, "last": "2026-11-05"}


def test_fail_line():
    r = {"ok": True, "held_out_check": "fail", "best_order": [2, 0, 1],
         "dm_best_vs_naive": {"p": 0.012}, "dm_best_vs_default": {"p": 0.2}}
    assert line("ABC", D, r) == ("ABC: fail best=(2, 0, 1) p_vs_naive=0.012 p_vs_101=0.200 "
                                 "rows=2979 last=2026-11-05")


def test_default_order_has_no_second_p():
    r = {"ok": True, "held_out_check": "pass", "best_order": [1, 0, 1],
         "dm_best_vs_naive": {"p": 0.5}, "dm_best_vs_default": None}
    assert "p_vs_101=-" in line("ABC", D, r)


def test_errors():
    bad = {"ok": False, "error": {"code": "BAD_TICKER"}}
    assert line("ZZZ", bad, {}) == "ZZZ: load_data BAD_TICKER"
    assert line("ABC", D, {"ok": False, "error": {"code": "OVER_BUDGET"}}) == "ABC: tune_arima OVER_BUDGET"


def test_parse_ends_and_tickers():
    a = parse(["--end", "2020-06-30", "2022-06-30", "--", "XOM", "CVX"])
    assert a.end == ["2020-06-30", "2022-06-30"]
    assert a.tickers == ["XOM", "CVX"]


def test_parse_default_end():
    a = parse(["XOM"])
    assert a.end == [END] and a.tickers == ["XOM"]
