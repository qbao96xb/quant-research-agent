# src/agent/loop.py
import json, time, uuid, re, threading
from src.tools.registry import schemas
from .skills import build_system_prompt
from .guardrails import safe_call
from .fit_output import fit_json
from .scope import cap_block, count_call
from .repeats import only_repeats, REPEAT_MSG
from .logging_utils import log_event

MAX_EMPTY_RETRIES = 2
MAX_INTERNAL_ERRORS = 2
MAX_DUPLICATES = 2          # CONSECUTIVE repeated calls before forcing a final answer
MAX_POLLS = 6               # consecutive "running" results before a nudge
TOOL_TIMEOUT = 300          # seconds per tool call (LLM timeout belongs in the HTTP client)
HISTORY_KEEP = 30
LOAD_INPUT_ERRORS = {"BAD_TICKER", "DATA_FETCH_FAILED", "BAD_DATE", "NOT_ENOUGH_DATA"}
NO_FALLBACK_TOOLS = {"list_artifacts"}   # saved data must not replace a failed load_data (case I01)
NO_FALLBACK_MSG = ("load_data failed earlier in this request, so saved data must not replace it. "
                   "Do not use another dataset or ticker. Tell the user what failed and ask for a "
                   "valid ticker or dates.")


def to_openai_tools(specs):
    out = []
    for s in specs:
        if s.get("type") == "function":
            out.append(s)
        else:
            out.append({"type": "function", "function": {
                "name": s["name"],
                "description": s.get("description", ""),
                "parameters": s.get("parameters")
                             or s.get("input_schema")
                             or {"type": "object", "properties": {}},
            }})
    return out


# ---------- answer cleaning (skips fenced code) ----------
_QUOTE_RE = re.compile(r"(?m)^[ \t]*(?:>[ \t]?)+")
_TABLE_RE = re.compile(r"(?m)([^\n|][^\n]*)\n(\|.*\|\s*\n\|[\s:|-]+\|)")
_FENCE_RE = re.compile(r"(`{3}.*?`{3})", re.S)


def clean_answer(text: str) -> str:
    """Strip blockquote markers and put a blank line before tables, outside code fences."""
    parts = _FENCE_RE.split(text or "")
    for i in range(0, len(parts), 2):          # even indexes = outside fences
        p = _QUOTE_RE.sub("", parts[i])
        parts[i] = _TABLE_RE.sub(r"\1\n\n\2", p)
    return "".join(parts).strip()


def _safe_emit(emit, *a, **k):
    try:
        emit(*a, **k)
    except Exception:          # a broken UI callback must not kill the run
        pass


# ---------- history ----------
def trim_history(messages, keep=HISTORY_KEEP, pin=None):
    """Keep system message + last `keep` messages. `pin` = index that must be kept
    (the current question), so a long run never loses its own turn."""
    if len(messages) <= keep + 1:
        return messages
    start = len(messages) - keep
    if pin is not None:
        start = min(start, pin)
    tail = messages[start:]
    while tail and tail[0].get("role") == "tool":   # never start on an orphan tool result
        tail = tail[1:]
    return [messages[0]] + tail


def _clean_msg(msg):
    """Keep only role/content/tool_calls; guarantee every tool call has an id."""
    out = {"role": "assistant", "content": msg.get("content") or ""}
    calls = [{**tc, "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"}
             for tc in (msg.get("tool_calls") or [])]
    if calls:
        out["tool_calls"] = calls
    return out


# ---------- tool execution with timeout ----------
def _call_tool(name, raw_args, timeout):
    box = {}

    def work():
        try:
            box["out"] = safe_call(name, raw_args)
        except Exception as e:
            box["out"] = ({"error": {"code": "INTERNAL", "message": str(e)[:200]}},
                          {"ok": False})

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():      # thread cannot be killed; it is abandoned
        return ({"error": {"code": "TIMEOUT",
                           "message": f"{name} exceeded {timeout:.0f}s"}},
                {"ok": False})
    return box["out"]


# ---------- duplicate-call cache ----------
def _key(name, raw_args):
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        return name, json.dumps(args, sort_keys=True)
    except Exception:
        return name, str(raw_args)


def _err_code(result):
    """Error code from nested {'error': {'code': ..}} or flat {'code': ..}."""
    if not isinstance(result, dict):
        return None
    err = result.get("error")
    if isinstance(err, dict):
        return err.get("code")
    return result.get("code")


def _is_running(result):
    """True if a tool result says a background job is still running."""
    return isinstance(result, dict) and result.get("status") == "running"


# ---------- grounding check ----------
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _nums(text):
    out = []
    for s in _NUM_RE.findall(text or ""):
        try:
            out.append(float(s.replace(",", "")))
        except ValueError:
            pass
    return out


def ungrounded(answer, messages):
    """Decimal / percent numbers in `answer` that appear in no tool output
    or user message. Bare integers (list numbers, years, '5-day') are skipped."""
    pool = []
    for m in messages:
        if m.get("role") in ("tool", "user"):
            c = m.get("content")
            if isinstance(c, str):
                pool += _nums(c)
    pool += [p * 100 for p in pool]            # fractions shown as percent
    bad = []
    for m in _NUM_RE.finditer(answer or ""):
        s = m.group().replace(",", "")
        after = (answer or "")[m.end():m.end() + 1]
        if "." not in s and after != "%":
            continue
        try:
            n = float(s)
        except ValueError:
            continue
        dec = len(s.split(".")[1]) if "." in s else 0
        tol = 0.5 * 10 ** (-dec) + 1e-6
        if not any(abs(n - p) <= tol for p in pool):
            bad.append(m.group() + ("%" if after == "%" else ""))
    return bad


def finalize(llm, messages, content, used_tools, run_id, step):
    """Grounding check for EVERY final answer.
    `content` is not yet in `messages`. Returns (answer, still_unverified)."""
    content = content or ""
    if not used_tools or not content.strip():
        return content, []
    bad = ungrounded(content, messages)
    if not bad:
        return content, []
    log_event(run_id=run_id, step=step, event="ungrounded", numbers=bad[:8])
    retry = trim_history(messages) + [
        {"role": "assistant", "content": content},
        {"role": "user", "content":
            f"These values are not in any tool output: {bad[:8]}. "
            "Edit your previous answer. Remove or replace only those values. "
            "Keep the same format and copy the *_pct fields from the tool results. "
            "Do not call any tool."}]
    try:
        new = (llm.chat(retry, None).get("content") or "").strip()
    except Exception as e:
        log_event(run_id=run_id, step=step, event="grounding_retry_failed",
                  error=str(e)[:200])
        new = ""
    if new:
        content = new
        bad = ungrounded(new, messages)   # pool excludes the retry prompt on purpose
    if bad:
        content += ("\n\nUnverified values (not found in tool output): "
                    + ", ".join(bad[:8]))
    return content, bad


def _force_final(llm, run_id, view_msgs, instruction):
    """One no-tools call. Returns text, or '' on failure."""
    try:
        msg = llm.chat(view_msgs + [{"role": "user", "content": instruction}], None)
        return msg.get("content") or ""
    except Exception as e:
        log_event(run_id=run_id, event="force_final_failed", error=str(e)[:200])
        return ""


def run_agent(question, llm, max_steps=12, run_timeout=900, on_event=None, history=None):
    raw_emit = on_event or (lambda *a, **k: None)
    emit = lambda *a, **k: _safe_emit(raw_emit, *a, **k)
    run_id = uuid.uuid4().hex[:8]
    t0 = time.time()
    tools = to_openai_tools(schemas())

    system = {"role": "system", "content": build_system_prompt()}
    messages = [_clean_msg(m) if m.get("role") == "assistant" else m
                for m in (history or [])]
    if messages and messages[0].get("role") == "system":
        messages[0] = system                     # refresh system prompt
    else:
        messages.insert(0, system)
    messages = trim_history(messages, HISTORY_KEEP)
    messages.append({"role": "user", "content": question})
    q_idx = len(messages) - 1

    def view():
        return trim_history(messages, HISTORY_KEEP, pin=q_idx)

    empty_retries = internal_errors = dup_count = poll_count = 0
    cache = {}
    used_tools = False
    last_hint = None
    failed_load = False
    done = {}

    def conclude(status, step, content):
        answer, bad = None, []
        if content and content.strip():
            answer, bad = finalize(llm, messages, content, used_tools, run_id, step)
            messages.append({"role": "assistant", "content": answer})
        return _end(run_id, status, step, messages, answer, t0=t0, unverified=bad)

    for step in range(1, max_steps + 1):
        if time.time() - t0 > run_timeout:
            return conclude("timeout", step - 1, None)
        emit("thinking", step=step, max_steps=max_steps)
        msg = llm.chat(view(), tools)
        content = msg.get("content") or ""
        calls = msg.get("tool_calls") or []
        log_event(run_id=run_id, step=step, event="llm_reply",
                  content=content[:500],
                  n_tool_calls=len(calls),
                  reasoning_len=len(msg.get("reasoning")
                                    or msg.get("reasoning_content") or ""),
                  finish_reason=msg.get("finish_reason"))

        if not calls and not content.strip():
            empty_retries += 1
            log_event(run_id=run_id, step=step, event="empty_reply",
                      retries=empty_retries)
            if empty_retries > MAX_EMPTY_RETRIES:
                return conclude("empty_reply", step, None)
            messages.append({"role": "user", "content":
                "Your last reply was empty. Continue with the next "
                "tool call, or give the final answer."})
            continue

        if not calls or only_repeats(calls, cache, _key, log_event, run_id, step):
            return conclude("done", step, content if not calls else _force_final(llm, run_id, view(), REPEAT_MSG))

        assistant = _clean_msg(msg)
        messages.append(assistant)

        for tc in assistant["tool_calls"]:
            fn = tc.get("function") or {}
            name = fn.get("name") or "unknown"
            raw_args = fn.get("arguments") or "{}"
            emit("tool_start", step=step, tool=name)
            t_call = time.time()

            key = _key(name, raw_args)
            if failed_load and name in NO_FALLBACK_TOOLS:
                result = {"ok": False, "error": {"code": "NO_FALLBACK", "message": NO_FALLBACK_MSG}}
                meta = {"ok": False, "error": "NO_FALLBACK", "blocked": True}
            elif key in cache:
                result, meta = cache[key]
                if not isinstance(result, dict):
                    result = {"result": result}
                result = {**result, "note": "Duplicate call. This is the previous "
                          "result. Do not call again, write the final answer."}
                meta = {**meta, "cached": True}
                dup_count += 1
            elif cap_block(name, done):
                result, meta = cap_block(name, done)
            else:
                dup_count = 0                       # only consecutive duplicates count
                budget = run_timeout - (time.time() - t0)
                result, meta = _call_tool(name, raw_args,
                                          max(5, min(TOOL_TIMEOUT, budget)))
                if meta.get("ok") and _is_running(result):
                    poll_count += 1
                    last_hint = result.get("stall_hint") or last_hint
                else:
                    poll_count = 0
                    if meta.get("ok"):              # cache finished successes only
                        cache[key] = (result, meta)
            used_tools = True
            count_call(name, result, meta, done)
            if name == "load_data" and not meta.get("ok") and _err_code(result) in LOAD_INPUT_ERRORS:
                failed_load = True
            dt = time.time() - t_call

            log_event(run_id=run_id, step=step, tool=name, **meta)
            emit("tool_end", step=step, tool=name, ok=meta.get("ok"), elapsed=dt)
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": fit_json(result)})
            if not meta.get("ok") and _err_code(result) in ("INTERNAL", "TIMEOUT"):
                internal_errors += 1

        if internal_errors >= MAX_INTERNAL_ERRORS:
            text = _force_final(llm, run_id, view(),
                "Tools failed twice (internal error or timeout). Stop calling tools. "
                "Give a short final answer saying what failed.")
            return conclude("internal_errors", step, text)

        if dup_count >= MAX_DUPLICATES:
            log_event(run_id=run_id, step=step, event="dup_escalation",
                      duplicates=dup_count)
            text = _force_final(llm, run_id, view(),
                "You are repeating calls. Stop calling tools and give the "
                "final answer from the results you already have.")
            return conclude("done", step, text)

        if poll_count >= MAX_POLLS:
            log_event(run_id=run_id, step=step, event="poll_nudge", polls=poll_count)
            messages.append({"role": "user", "content": last_hint or
                "A background job is still running after several polls. Tell the "
                "user it is still running and give its id."})
            poll_count = 0

    text = _force_final(llm, run_id, view(),
        "Step limit reached. Do not call tools. Summarize what you found so far "
        "and what is still missing.")
    return conclude("max_steps", max_steps, text)


def _end(run_id, status, steps, messages, answer=None, t0=None, unverified=None):
    if answer is None:
        answer = f"Stopped early ({status}). See the log for run {run_id}."
    else:
        answer = clean_answer(answer)
    elapsed = round(time.time() - t0, 1) if t0 else None
    log_event(run_id=run_id, event="run_end", status=status,
              steps=steps, elapsed=elapsed, unverified=unverified or [])
    return dict(run_id=run_id, status=status, steps=steps, answer=answer,
                messages=messages,                    # full, for eval traces
                history=trim_history(messages),       # bounded, save this one
                unverified=unverified or [])