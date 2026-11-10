import json, re
from pathlib import Path
from .number_check import check_numbers

PROMPT = (Path(__file__).resolve().parents[1] / "agent" / "prompts" / "narrative.md").read_text()

def slim(f):
    keep = ("ticker", "h", "describe", "forecast", "metrics", "dm", "garch", "verdict")
    return {k: f[k] for k in keep}

def parse(text):
    parts = re.split(r"###\s*(SUMMARY|INTERPRETATION|CAVEATS)\s*", text)
    d = {parts[i].lower(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    if set(d) != {"summary", "interpretation", "caveats"}:
        raise ValueError("missing sections")
    return d

def fallback(f):
    m = next(iter(f["verdict"]))
    r = f["forecast"][m]
    return dict(status="fallback",
                summary=f"{f['ticker']} {f['h']}-day {m} forecast: {r['mean']:.4f}, 90% interval [{r['lo']:.4f}, {r['hi']:.4f}].",
                interpretation="\n".join(f"- {k}: {v}" for k, v in f["verdict"].items()),
                caveats="- See fixed caveats below.")

def write_narrative(f, llm, retries=1):
    msgs = [{"role": "system", "content": PROMPT},
            {"role": "user", "content": "FACTS:\n" + json.dumps(slim(f), default=str)}]
    for attempt in range(retries + 1):
        out = llm.chat(msgs, tools=[])["content"]
        try:
            text = parse(out)
        except ValueError as e:
            bad = [str(e)]
        else:
            bad = check_numbers(" ".join(text.values()), f)
            if not bad and not wording_violation(text, f):
                return {**text, "status": "llm"}
            bad = bad or ["wording"]
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": f"Rejected: {bad}. Rewrite using only numbers from FACTS and the verdict wording."}]
    return fallback(f)

def wording_violation(text, f):
    flat = " ".join(text.values()).lower()
    no_sig = any("no significant" in v.lower() for v in f["verdict"].values())
    return no_sig and bool(re.search(r"\b(outperform\w*|beats?|superior|significantly better)\b", flat))