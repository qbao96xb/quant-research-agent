import re

NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
MODEL = re.compile(r"(GARCH|ARIMA|ARMA)\([^)]*\)", re.I)

def flatten(x, out=None):
    out = [] if out is None else out
    if isinstance(x, dict):
        for v in x.values(): flatten(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x: flatten(v, out)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out.append(float(x))
    return out

def decimals(tok):
    s = tok.rstrip("%").replace(",", "")
    return len(s.split(".")[1]) if "." in s else 0

def matches(tok, pool):
    s = tok.rstrip("%").replace(",", ""); v = float(s); d = decimals(tok)
    tol = 0.5 * 10 ** (-d) + 1e-12
    cands = [p * 100 for p in pool] if tok.endswith("%") else pool + [p * 100 for p in pool]
    return any(abs(round(c, d) - v) <= tol or abs(c - v) <= tol for c in cands)

def check_numbers(text, facts, extra=(0.05, 0.9, 90, 100)):
    pool = flatten(facts) + [float(facts["h"])] + list(extra)
    clean = MODEL.sub("", DATE.sub("", text))
    clean = re.sub(r"(?m)^\s*\d+\.\s", "", clean)
    bad = []
    for tok in NUM.findall(clean):
        if tok.lstrip("+-").isdigit() and len(tok.lstrip("+-")) == 4 and 1900 <= int(tok) <= 2100:
            continue
        if not matches(tok, pool):
            bad.append(tok)
    return sorted(set(bad))