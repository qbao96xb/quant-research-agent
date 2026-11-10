import pytest
from src.report.narrative import parse, write_narrative, wording_violation

def good(f):
    m = next(iter(f["forecast"]))
    r = f["forecast"][m]
    v = next(iter(f["verdict"].values()))
    return (f"### SUMMARY\n{m} forecast {r['mean']:.4f}. {v}.\n"
            f"### INTERPRETATION\n- ok\n### CAVEATS\n- small sample")

BAD = ("### SUMMARY\nForecast 0.9999.\n### INTERPRETATION\n- x\n### CAVEATS\n- y")

def test_parse_missing_heading():
    with pytest.raises(ValueError):
        parse("### SUMMARY\nonly this")

def test_wording_violation(facts):
    facts["verdict"] = {"arima": "No significant difference from naive (DM p=0.333)"}
    t = {"summary": "ARIMA outperforms naive", "interpretation": "", "caveats": ""}
    assert wording_violation(t, facts)

def test_llm_ok(facts, scripted):
    out = write_narrative(facts, scripted([good(facts)]))
    assert out["status"] == "llm"

def test_retry_then_ok(facts, scripted):
    llm = scripted([BAD, good(facts)])
    assert write_narrative(facts, llm)["status"] == "llm"
    assert llm.calls == 2

def test_two_bad_fallback(facts, scripted):
    assert write_narrative(facts, scripted([BAD, BAD]))["status"] == "fallback"