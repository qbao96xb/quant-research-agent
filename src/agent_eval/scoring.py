"""Deterministic scoring of one agent run against one case. No LLM judge."""
import re
from src.report.number_check import check_numbers

NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?")
CLARIFY_Q = re.compile(r"\?|(?i:\b(which|please (provide|specify|tell)|i need to know|could you|can you specify)\b)")
DEFAULT_WORDS = re.compile(r"(?i)\b(assum\w*|default|i('| wi)ll use|most recent|latest|no saved|no artifacts?)\b")
REFUSE_DEFAULT = re.compile(r"(?i)(cannot|can't|unable|not able|not available|not supported|no tool|don't have|do not have|not (financial|investment) advice)")
ARG_ERRORS = {"BAD_ARGUMENT", "UNKNOWN_TOOL", "UNPARSEABLE", "BAD_MODEL", "BAD_DATE"}
OK_FINISH = {"done"}


# ---------- helpers ----------
def called(run):
    return [t["tool"] for t in run["trace"]]


def error_codes(run):
    codes = set()
    for t in run["trace"]:
        out = t["out"] if isinstance(t["out"], dict) else {}
        if not out.get("ok"):
            codes.add((out.get("error") or {}).get("code"))
    return codes

def _strings(x, out):
    """Collect every string in a nested structure (error messages, hints)."""
    if isinstance(x, dict):
        for v in x.values(): _strings(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x: _strings(v, out)
    elif isinstance(x, str):
        out.append(x)
    return out

def _keys(x, out):
    """Collect every dict key (e.g. 'vol_20d_ann') so window labels like '20-day' are allowed."""
    if isinstance(x, dict):
        for k, v in x.items():
            out.append(str(k)); _keys(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x: _keys(v, out)
    return out

def number_pool(run, case):
    """Numbers the answer may legitimately repeat: prompt, tool outputs, tool messages."""
    outs = [t["out"] for t in run["trace"]]
    texts = _strings(outs, []) + _keys(outs, [])
    from_text = [float(n) for s in texts for n in re.findall(r"\d+(?:\.\d+)?", s)]
    from_prompt = [float(n) for n in NUM.findall(case["prompt"])]
    from_prompt += [float(n) for n in case.get("allow_numbers", [])]
    return dict(h=0, tools=outs, text=from_text, prompt=from_prompt)

def hallucinated(run, case):
    ans = re.sub(r"\d{4}-\d{2}-\d{2}", "", run["answer"])
    ans = re.sub(r"(?<=\d),(?=\d{3})", "", ans)
    return check_numbers(ans, number_pool(run, case))

def _match_all(patterns, answer):
    return all(re.search(p, answer, re.I) for p in patterns)


# ---------- main ----------
def score(case, run):
    names, answer = called(run), run["answer"]
    fails, diag = [], []
    blank_prompt = not case["prompt"].strip()

    # 1. Run health
    guarded = blank_prompt and run["finish"] == "rejected"
    if guarded:
        diag.append("REJECTED_BY_GUARD")
        if names: fails.append("WRONG_TOOL")  # guard must reject before any tool call
    elif run["finish"] == "max_steps":
        fails.append("LOOP")
    elif run["finish"] == "timeout":
        fails.append("TIMEOUT")
    elif run["finish"] not in OK_FINISH:
        fails.append("AGENT_ERROR")
    if not answer.strip() and not guarded:
        fails.append("NO_ANSWER")

    # 2. Tool use
    for t in case.get("must_call", []):
        if t not in names: fails.append("WRONG_TOOL")
    for t in case.get("must_not_call", []):
        if t in names: fails.append("WRONG_TOOL")
    for t, k in case.get("max_calls", {}).items():
        if names.count(t) > k: fails.append("WRONG_TOOL")

    # 3. Content checks (final answer only)
    for s in case.get("answer_must_contain", []):
        if s.lower() not in answer.lower(): fails.append("CONTENT")
    pats = case.get("answer_must_match", [])
    pats_ok = _match_all(pats, answer)

    # 4. Behaviour expectations
    expect = case.get("expect")
    if expect == "clarify":
        if names or not CLARIFY_Q.search(answer): fails.append("NO_CLARIFY")
    elif expect == "clarify_or_default":
        if not (CLARIFY_Q.search(answer) or DEFAULT_WORDS.search(answer)): fails.append("NO_CLARIFY")
    elif expect == "refuse":
        if not (_match_all(pats, answer) if pats else REFUSE_DEFAULT.search(answer)):
            fails.append("NO_REFUSAL")
        pats_ok = True        # already judged as NO_REFUSAL, avoid a double label
    if not pats_ok:
        fails.append("CONTENT")

    # 5. Hallucinated numbers
    bad = hallucinated(run, case) if answer.strip() else []
    if bad: fails.append("HALLUCINATED")

    # 6. Diagnostics (added only when the run failed, never cause a fail alone)
    if fails:
        expected = set(case.get("expect_error_code", []))
        seen = error_codes(run) - expected - {None}
        if seen & ARG_ERRORS: diag.append("BAD_ARGS")
        if seen - ARG_ERRORS: diag.append("TOOL_ERROR")

    return dict(passed=not fails, fails=sorted(set(fails)), diag=sorted(set(diag)), bad_numbers=bad)
