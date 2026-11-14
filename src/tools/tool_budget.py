"""Work budgets for slow tools (owner, 2026-11-13): refuse oversized requests at once, keep runs fast."""
from .errors import ToolError

FIT_BUDGET = 1200        # ARIMA fits per tune_arima call, about 80 s
SEC_PER_FIT = 0.07       # measured 2026-11-13: defaults (620 fits) take 40 to 45 s


def tune_fits(max_p, max_q, n_val, n_test):
    """Fits one tune_arima call needs: every order on n_val origins, plus 2 x n_test on the test block."""
    return (max_p + 1) * (max_q + 1) * n_val + 2 * n_test


def check_tune_budget(args):
    """Raise OVER_BUDGET when the request needs more than FIT_BUDGET fits; return the fit count."""
    fits = tune_fits(args.max_p, args.max_q, args.n_val, args.n_test)
    if fits > FIT_BUDGET:
        raise ToolError(
            "OVER_BUDGET",
            f"This tuning needs about {fits} model fits (about {round(fits * SEC_PER_FIT)} s); "
            f"the limit is {FIT_BUDGET} fits.",
            "Use the defaults (max_p=2, max_q=2, n_val=60, n_test=40), or a smaller grid or n_val.")
    return fits