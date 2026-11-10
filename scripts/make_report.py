import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.agent.llm import OpenAICompatLLM
from src.report.pipeline import make_report

ticker = sys.argv[1] if len(sys.argv) > 1 else "KO"
no_llm = "--no-llm" in sys.argv
p = make_report(ticker, None if no_llm else OpenAICompatLLM())
print(f"Report written: {p}")
