from pathlib import Path

AGENT = Path(__file__).resolve().parent

def build_system_prompt() -> str:
    base = (AGENT / "prompts" / "system.md").read_text()
    skills = [p.read_text() for p in sorted((AGENT / "skills").glob("*.md"))]
    return base + "\n\n# Skills\n" + "\n\n".join(skills)