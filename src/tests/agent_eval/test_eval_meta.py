"""Eval runs record the git commit, branch and dirty flag they tested (E1)."""
import subprocess

import scripts.eval_agent as eval_agent


def test_git_info_matches_git():
    info = eval_agent.git_info()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert set(info) == {"commit", "branch", "dirty"}
    assert info["commit"] == head
    assert isinstance(info["dirty"], bool)


def test_git_info_outside_a_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert eval_agent.git_info() == {"commit": "", "branch": "", "dirty": False}
