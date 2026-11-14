import json, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
from src.tools.registry import call, REGISTRY

# TRANSIENT = {"DATA_FETCH_FAILED", "TIMEOUT"}
TRANSIENT = {"DATA_FETCH_FAILED"}
_POOL = ThreadPoolExecutor(max_workers=2)

def truncate(s, n=4000):
    return s if len(s) <= n else s[:n] + '..."truncated"'

def _err(code, msg):
    return {"ok": False, "error": {"code": code, "message": msg}}

def _run(name, args, timeout):
    fut = _POOL.submit(call, name, **args)
    try:
        return fut.result(timeout=timeout)
    except FTimeout:
        return _err("TIMEOUT", f"{name} exceeded {timeout}s")
    except Exception as e:
        return _err("TOOL_CRASH", f"{type(e).__name__}: {e}")

def safe_call(name, raw_args, timeout=300):   # = loop.TOOL_TIMEOUT (test_tool_timeout.py)
    t0 = time.time()
    if name not in REGISTRY:
        r = _err("UNKNOWN_TOOL", f"{name} not in {sorted(REGISTRY)}")
        return r, _meta(name, {}, r, t0, 0)
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except json.JSONDecodeError as e:
        r = _err("BAD_ARGUMENT", f"Arguments are not valid JSON: {e}")
        return r, _meta(name, {}, r, t0, 0)
    r = _run(name, args, timeout)
    retries = 0
    if not r.get("ok") and r.get("error", {}).get("code") in TRANSIENT:
        retries = 1
        r = _run(name, args, timeout)
    return r, _meta(name, args, r, t0, retries)

def _meta(name, args, r, t0, retries):
    return dict(args=args, ok=bool(r.get("ok")),
                error=(r.get("error") or {}).get("code"),
                retries=retries, ms=int((time.time() - t0) * 1000))