import argparse
import json
import re
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

ROOT = Path(__file__).resolve().parent.parent
# ROOT/src is still needed: eval_tools.py imports `backtest.jobs` without the `src.` prefix.
# Remove it only after those imports are fixed.
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from src.agent.llm import OpenAICompatLLM
from src.agent.loop import run_agent
from src.agent.session import load_history, save_history
from src.agent.findings import write_findings

DEFAULT_QUESTION = "List my saved artifacts"
SESSION_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def session_name(value):
    if not SESSION_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "use 1-64 characters: letters, digits, '_' or '-'")
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="run_agent.py",
        description="Ask the local quant research agent a question.",
        epilog='Example: python scripts/run_agent.py --session demo "Backtest KO with arima and naive"',
    )
    p.add_argument("question", nargs="*",
                   help=f'Question for the agent (default: "{DEFAULT_QUESTION}"). '
                        "If omitted and stdin is piped, the question is read from stdin.")
    p.add_argument("--session", metavar="NAME", type=session_name,
                   help="Session name. Loads and saves history at artifacts/sessions/NAME.json")
    p.add_argument("--findings", action="store_true",
                   help="After a successful run, run write_findings.py to draft results/findings.md")
    p.add_argument("--max-steps", type=int, default=12, metavar="N",
                   help="Maximum model steps (default: 12)")
    p.add_argument("--timeout", type=int, default=900, metavar="SEC",
                   help="Run timeout in seconds (default: 900)")
    p.add_argument("--json", action="store_true",
                   help="Print a JSON result on stdout instead of formatted Markdown")
    return p.parse_args(argv)


def make_logger():
    """Progress logger writing to stderr, with timings relative to creation."""
    t0 = time.time()

    def show(event, **k):
        t = f"{time.time() - t0:5.1f}s"
        if event == "thinking":
            msg = f"step {k.get('step')}/{k.get('max_steps')}: model thinking..."
        elif event == "tool_start":
            msg = f"  -> calling {k.get('tool')}"
        elif event == "tool_end":
            status = "ok" if k.get("ok") in (True, None) else "FAILED"
            msg = f"  <- {k.get('tool')} {status} ({k.get('elapsed', 0):.1f}s)"
        else:
            return
        print(f"[{t}] {msg}", file=sys.stderr, flush=True)

    return show


def read_question(a):
    question = " ".join(a.question).strip()
    if not question and not sys.stdin.isatty():
        question = sys.stdin.read().strip()
    return question or DEFAULT_QUESTION


def main():
    a = parse_args()
    session = a.session
    question = read_question(a)
    history = load_history(session)

    try:
        r = run_agent(question, OpenAICompatLLM(), max_steps=a.max_steps,
                      run_timeout=a.timeout, on_event=make_logger(), history=history)
    except KeyboardInterrupt:
        print("\nInterrupted. Session not saved.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        # run_agent raised, so there is no new history to save. The old session file is untouched.
        print(f"\nAgent failed: {type(e).__name__}: {e}\nSession not updated.",
              file=sys.stderr)
        sys.exit(1)

    save_history(session, r["history"])          # trimmed history, not the full trace

    ok = r["status"] == "done"
    if a.json:
        print(json.dumps({k: r.get(k) for k in
                          ("run_id", "status", "steps", "answer", "unverified")}))
    else:
        suffix = f" session={session}" if session else ""
        print(f"\n[{r['status']}] run={r['run_id']} steps={r['steps']}{suffix}\n")
        Console().print(Markdown(r["answer"]))

    if a.findings:
        if ok:
            md, _, charts = write_findings(r, question)
            print(f"\nFindings: {md} ({len(charts)} charts)", file=sys.stderr)
        else:
            print(f"\nFindings skipped: run status is '{r['status']}'.", file=sys.stderr)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()