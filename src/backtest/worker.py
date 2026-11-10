import json
import sys
import traceback

from backtest.jobs import write_state
from tools.eval_tools import run_backtest_blocking

def main(bid, args_json):
    args = json.loads(args_json)
    try:
        run_backtest_blocking(bid, args,
                              on_model_done=lambda i: write_state(bid, done=i))
        write_state(bid, status="done", done=len(args["models"]))
    except Exception:
        write_state(bid, status="failed", error=traceback.format_exc()[-500:])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
