import json
from pathlib import Path
import pytest

FIX = Path(__file__).parent / "fixtures" / "facts_ko.json"

@pytest.fixture
def facts():
    return json.loads(FIX.read_text())

class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
    def chat(self, messages, tools=None):
        self.calls += 1
        return {"content": self.replies.pop(0)}

@pytest.fixture
def scripted():
    return ScriptedLLM