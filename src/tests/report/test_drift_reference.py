import pytest

from src.report.facts import collect_facts
from src.report.narrative import slim
from src.report.number_check import check_numbers


@pytest.fixture(scope="module")
def ko():
    return collect_facts("KO", start="2015-01-01")


def test_dm_drift_only_for_other_models(ko):
    assert set(ko["dm_drift"]) == {"arima"}
    p = ko["dm_drift"]["arima"]["p_value"]
    assert 0.0 <= p <= 1.0


def test_verdicts_name_their_benchmark(ko):
    assert "from drift (reference)" in ko["verdict_drift"]["arima"]
    assert "from naive" in ko["verdict"]["arima"]
    assert "drift" not in ko["verdict"]["arima"]


def test_slim_keeps_drift_facts_and_tolerates_missing(ko, facts):
    s = slim(ko)
    assert "dm_drift" in s and "verdict_drift" in s
    assert "dm_drift" not in slim(facts)


def test_drift_p_value_passes_number_check(ko):
    p = ko["dm_drift"]["arima"]["p_value"]
    assert check_numbers(f"Against drift the DM p-value is {p:.3f}.", ko) == []
