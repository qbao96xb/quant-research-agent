# src/tools/tune_arima_tools.py
from evaluation.tuning import SEARCH_CFG, tune_arima as _tune

from .cache import clean_json, data_fp, get_dataset, put
from .errors import ToolError
from .registry_core import tool
from .schemas import TuneArimaArgs


@tool(TuneArimaArgs, name="tune_arima")
def tune_arima(args: TuneArimaArgs):
    """Select the best ARIMA order (p, 0, q) for a dataset, without look-ahead.

    Fits every order on a validation block, ranks them by out-of-sample RMSE,
    then checks the winner on a held-out test block against naive and ARIMA(1,0,1).
    Returns best_order, a ranked table (paste verbatim), test RMSE, DM p-values
    and a note. Use best_order in forecast(order=...). Never pick an order yourself.
    """
    df = get_dataset(args.dataset_id)

    try:
        out = _tune(df, h=args.h, max_p=args.max_p, max_q=args.max_q,
                    n_val=args.n_val, n_test=args.n_test)
    except ValueError as e:
        if str(e).startswith("NOT_ENOUGH_DATA"):
            raise ToolError(
                "NOT_ENOUGH_DATA", str(e),
                "Load an earlier start date, or lower n_val and n_test.") from e
        raise

    payload = dict(ds=args.dataset_id, fp=data_fp(df), h=args.h,
                   max_p=args.max_p, max_q=args.max_q,
                   n_val=args.n_val, n_test=args.n_test, cfg=SEARCH_CFG, v=1)
    out = clean_json(out)
    out["tune_id"] = put("tune", out, payload)
    out["ok"] = True
    return out
