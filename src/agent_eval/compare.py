"""Run the eval for every model in MODELS and compare.

PYTHONPATH=src:. python -m src.agent_eval.compare --split dev --n 3
"""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.eval_agent import evaluate
from src.agent.llm import OpenAICompatLLM

MODELS = {
    "qwen3.5-9b": "qwen3.5:9b",
    "qwen2.5-coder-14b": "qwen2.5-coder:14b",
}   # name -> your config


def main(n=3, prompt_ver="v2", split="dev"):
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    rows, cats, errors = [], {}, {}
    for name, cfg in MODELS.items():
        out = f"results/eval/{name}_{prompt_ver}_{split}_{stamp}"
        try:
            m, _, tables = evaluate(OpenAICompatLLM(model=cfg), out, n=n, split=split,
                                    meta=dict(name=name, model=cfg, n=n, split=split,
                                              prompt_ver=prompt_ver))
        except Exception as e:                      # one broken model must not stop the rest
            errors[name] = repr(e)
            print(f"[{name}] FAILED: {e!r}")
            continue
        rows.append(dict(model=name, **m))
        cats[name] = tables["by_category"]

    root = Path("results/eval")
    root.mkdir(parents=True, exist_ok=True)
    if rows:
        t = pd.DataFrame(rows).round(3)
        c = pd.DataFrame(cats).T.round(3)           # model x category pass rate
        t.to_csv(root / f"compare_{prompt_ver}_{split}_{stamp}.csv", index=False)
        c.to_csv(root / f"compare_cat_{prompt_ver}_{split}_{stamp}.csv")
        print("\n== Totals ==\n" + t.to_markdown(index=False))
        print("\n== Pass rate by category ==\n" + c.to_markdown())
    for k, v in errors.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--prompt-ver", default="v2")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    a = ap.parse_args()
    main(n=a.n, prompt_ver=a.prompt_ver, split=a.split)
