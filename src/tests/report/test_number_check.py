from src.report.number_check import check_numbers

F = {"h": 5, "x": 0.00304, "y": 0.01903, "n": 99}

def test_fact_passes():
    assert check_numbers("Value is 0.019.", F) == []

def test_percent_scaling_passes():
    assert check_numbers("Forecast is 0.30%.", F) == []

def test_wrong_percent_fails():
    assert check_numbers("Forecast is 0.03%.", F) == ["0.03%"]

def test_invented_number_flagged():
    assert "0.0421" in check_numbers("Error was 0.0421.", F)

def test_ignored_tokens():
    txt = "In 2024-05-01 and 2015, GARCH(1,1) gave:\n1. a result"
    assert check_numbers(txt, F) == []

def test_whitelist():
    assert check_numbers("90% interval, p above 0.05.", F) == []