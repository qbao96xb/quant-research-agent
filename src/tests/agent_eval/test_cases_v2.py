import re

import yaml

from src.agent_eval.paths import CASES_PATH, CASES_V2_PATH
from src.tools.registry import REGISTRY


def load(p):
    return yaml.safe_load(p.read_text())


def test_v2_ids_unique_and_new():
    v1 = {c["id"] for c in load(CASES_PATH)}
    ids = [c["id"] for c in load(CASES_V2_PATH)]
    assert ids == ["T01", "T02", "C01", "C02", "C03", "C04"]
    assert not set(ids) & v1


def test_v2_dev_split_only():
    assert all(c["split"] == "dev" for c in load(CASES_V2_PATH))


def test_v2_tool_names_exist():
    for c in load(CASES_V2_PATH):
        names = c.get("must_call", []) + c.get("must_not_call", []) + list(c.get("max_calls", {}))
        assert set(names) <= set(REGISTRY), c["id"]


def test_t02_patterns_match_fail_question():
    t02 = [c for c in load(CASES_V2_PATH) if c["id"] == "T02"][0]
    answer = "Please choose how to proceed: 1) keep (0,0,1), 2) fall back to the default ARIMA(1,0,1), 3) re-tune on a different split"
    for p in t02["answer_must_match"]:
        assert re.search(p, answer), p
