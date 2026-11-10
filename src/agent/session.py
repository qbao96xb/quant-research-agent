import json
import os
import re
import tempfile
from pathlib import Path

SESS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "sessions"
_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _path(name: str) -> Path:
    name = (name or "").strip()
    if not _NAME_RE.fullmatch(name):
        raise ValueError("Invalid session name: use 1-64 letters, digits, '_' or '-'")
    return SESS_DIR / f"{name}.json"


def load_history(name: str | None):
    """Return saved messages for a session, or None if new or corrupt."""
    if not name:
        return None
    p = _path(name)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("session file is not a list")
        return data
    except (ValueError, UnicodeDecodeError):      # JSONDecodeError is a ValueError
        p.replace(p.with_suffix(".json.bak"))     # keep the corrupt file, start fresh
        return None


def save_history(name: str | None, messages) -> None:
    """Atomically save messages for a session."""
    if not name:
        return
    p = _path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(messages, f, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
