import json
import threading
import time
from pathlib import Path

import yaml

from src.agent.loop import run_agent          # adjust import

from src.agent_eval.paths import CASES_PATH
TIMEOUT_S = 300

def tool_trace(messages):
    calls, trace = {}, []
    for m in messages:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                calls[tc["id"]] = (tc["function"]["name"], tc["function"].get("arguments"))
        elif m["role"] == "tool":
            try:
                out = json.loads(m["content"])
            except Exception:
                out = {"ok": False, "error": {"code": "UNPARSEABLE"}}
            name, args = calls.get(m.get("tool_call_id"), ("?", None))
            trace.append(dict(tool=name, args=args, out=out))
    return trace


def _call_agent(case, llm, box):
    try:
        box["res"] = run_agent(case["prompt"], llm=llm, max_steps=case.get("max_steps", 12))
    except ValueError as e:                       # adjust to the exception your guard raises
        box["res"] = dict(answer="", messages=[], steps=0, finish="rejected")
        box["err"] = repr(e)
    except Exception as e:
        box["res"] = dict(answer="", messages=[], steps=0, finish="error")
        box["err"] = repr(e)


def run_case(case, llm, run_idx, timeout=TIMEOUT_S):
    box, t0 = {}, time.perf_counter()
    th = threading.Thread(target=_call_agent, args=(case, llm, box), daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():                             # thread cannot be killed, it is abandoned
        res, err = dict(answer="", messages=[], steps=0, finish="timeout"), "timeout"
    else:
        res, err = box.get("res") or dict(answer="", messages=[], steps=0, finish="error"), box.get("err")
    return dict(case_id=case["id"], category=case["category"], run=run_idx,
                latency=time.perf_counter() - t0,
                answer=res.get("answer") or "", steps=res.get("steps", 0),
                finish=res.get("finish") or res.get("status") or "error",
                error=err,
                trace=tool_trace(res.get("messages", [])))

def warm_up(llm):
    try:
        llm.chat([{"role": "user", "content": "Say ok."}], tools=[])
    except Exception:
        pass

def run_all(llm, out_dir, n=3, only=None, cases_path=CASES_PATH, resume=False, on_event=None):
    """on_event(kind, **info): kinds are 'start', 'case_start', 'run_done'."""
    emit = on_event or (lambda *a, **k: None)
    cases = yaml.safe_load(Path(cases_path).read_text())
    cases = [c for c in cases if not only or c["id"] in only]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "runs.jsonl"

    done = set()
    if path.exists():
        if resume:
            done = {(r["case_id"], r["run"]) for r in map(json.loads, path.read_text().splitlines())}
        else:
            raise FileExistsError(f"{path} exists. Use resume=True or a new folder.")

    ids = {c["id"] for c in cases}
    emit("start", total=len(cases) * n,
         done=sum(1 for cid, i in done if cid in ids and i < n))
    warm_up(llm)
    with open(path, "a") as fh:
        for c in cases:
            emit("case_start", case=c, n=n,
                 done=sum((c["id"], i) in done for i in range(n)))
            for i in range(n):
                if (c["id"], i) in done:
                    continue
                rec = run_case(c, llm, i)
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                emit("run_done", case=c, run=rec)
