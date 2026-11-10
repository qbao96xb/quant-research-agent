from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "configs" / "daily_small.template.yaml"
dst = ROOT / "configs" / "daily_small.yaml"

text = src.read_text(encoding="utf-8")
if "{ROOT}" not in text:
    raise ValueError("No {ROOT} placeholder found in template")

out = text.replace("{ROOT}", ROOT.as_posix())
if "{" in out and "}" in out:
    print("Warning: unresolved placeholder(s) may remain")

dst.write_text(out, encoding="utf-8")
print("Wrote", dst)
