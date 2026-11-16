"""The system prompt tells the agent to refuse look-ahead requests in words (case C04, owner 2026-11-14)."""
from pathlib import Path

SYSTEM_MD = Path(__file__).parents[2] / "agent" / "prompts" / "system.md"


def test_lookahead_rule_in_out_of_scope():
    text = SYSTEM_MD.read_text()
    section = text.split("# Out of scope", 1)[1].split("\n# ", 1)[0]
    assert "look-ahead bias and would make the results invalid" in section
    assert "walk-forward version" in section
