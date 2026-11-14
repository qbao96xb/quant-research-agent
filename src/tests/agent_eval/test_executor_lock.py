"""evaluate() refuses to start while an executor run holds handoff/.executor.lock."""
import pytest

import scripts.eval_agent as eval_agent


def test_evaluate_refuses_while_executor_lock_exists(tmp_path, monkeypatch):
    lock = tmp_path / ".executor.lock"
    lock.mkdir()
    monkeypatch.setattr(eval_agent, "EXECUTOR_LOCK", lock)
    with pytest.raises(SystemExit, match="executor run active"):
        eval_agent.evaluate(llm=None, out=str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_lock_path_points_at_repo_handoff():
    assert eval_agent.EXECUTOR_LOCK.parts[-2:] == ("handoff", ".executor.lock")
