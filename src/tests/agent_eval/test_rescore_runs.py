"""scripts/rescore_runs.py re-scores saved runs and reports changed results; it writes nothing."""
import json

from scripts.rescore_runs import main, rescore

CASES_YAML = "- id: T02\n  category: tuning\n  prompt: Tune ARIMA\n  max_calls: {tune_arima: 1}\n"
REFUSED = dict(tool="tune_arima", out={"ok": False, "error": {"code": "OVER_BUDGET"}})
OK = dict(tool="tune_arima", out={"ok": True})


def write_run(d, trace, passed):
    run = dict(case_id="T02", category="tuning", run=0, answer="ok", finish="done",
               steps=3, latency=1.0, trace=trace)
    (d / "runs.jsonl").write_text(json.dumps(run) + "\n")
    (d / "scored.csv").write_text(f"case,cat,split,run,passed\nT02,tuning,dev,0,{passed}\n")


def test_refused_retry_now_passes(tmp_path):
    write_run(tmp_path, [REFUSED, OK], False)
    cases = {"T02": dict(id="T02", category="tuning", prompt="Tune ARIMA", max_calls={"tune_arima": 1})}
    assert rescore(tmp_path, cases) == [("T02", 0, False, True, "")]


def test_main_prints_change_and_writes_nothing(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_run(run_dir, [REFUSED, OK], False)
    cases = tmp_path / "cases.yaml"
    cases.write_text(CASES_YAML)
    before = sorted(p.name for p in run_dir.iterdir())
    assert main([str(run_dir), "--cases", str(cases)]) == 0
    out = capsys.readouterr().out
    assert "old 0/1  new 1/1  changed 1" in out
    assert "T02 run 0: fail -> pass" in out
    assert sorted(p.name for p in run_dir.iterdir()) == before
