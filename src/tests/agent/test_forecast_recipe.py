"""Only daily wording selects the forecast_path recipe; "forecast X for N days" is forecast only (S03, 2026-11-15)."""
from pathlib import Path

SYSTEM_MD = Path(__file__).parents[2] / "agent" / "prompts" / "system.md"


def workflows():
    return SYSTEM_MD.read_text().split("# Workflows", 1)[1].split("\n# ", 1)[0]


def test_path_recipe_needs_daily_words():
    w = workflows()
    assert "Forecast for the next N days" not in w
    assert "- Daily forecast path, only when the user asks for" in w


def test_forecast_only_names_n_days():
    w = workflows()
    assert "- Forecast only, including \"forecast X for N days\"" in w
    assert "call forecast once with h=N" in w
