# src/report/builder.py
from datetime import date
from .tables import metrics_table, forecast_table, data_table, garch_table

SLOTS = ("summary", "interpretation", "caveats")


def img(figs, key, alt):
    """Markdown image, or a visible note if the chart failed."""
    p = figs.get(key, "MISSING (not built)")
    if isinstance(p, (list, tuple)):
        return "\n\n".join(f"![{alt}]({q})" for q in p) or f"*Chart unavailable: {key}*"
    if str(p).startswith("MISSING"):
        return f"*Chart unavailable: {p}*"
    return f"![{alt}]({p})"


def _section(title, *parts):
    body = "\n\n".join(str(p).strip() for p in parts if p and str(p).strip())
    return f"## {title}\n\n{body}"


def _fixed_caveats(f):
    n = f.get("describe", {}).get("n", "?")
    caveats = [
        f"Data n={n}, forecast horizon h={f['h']}; targets overlap when step < h.",
        "Accuracy measures forecast error, not trading profit.",
        "Research only, not investment advice.",
    ]
    if f.get("n_origins"):
        caveats.insert(1, f"Backtest origins: {f['n_origins']}.")
    return caveats


def _bullets(items):
    return "\n".join(f"- {i}" for i in items)


def build_markdown(f, figs, text):
    txt = {k: text.get(k) or "_Not available._" for k in SLOTS}
    verdicts = _bullets(f"**{m}:** {v}" for m, v in f["verdict"].items())
    last_date = f.get("describe", {}).get("end") or f.get("describe", {}).get("last_date") or "?"

    appendix = _bullets([
        f"ticker: `{f['ticker']}`",
        f"data start: {f.get('start', '?')}, last data date: {last_date}",
        f"horizon h: {f['h']}",
        f"models: {', '.join(f['metrics'])}",
        f"dataset_id: `{f['dataset_id']}`",
        f"backtest_id: `{f['backtest_id']}`",
        f"report date: {date.today():%Y-%m-%d}",
        f"narrative: {text.get('status', 'llm')}",
    ])

    return "\n\n".join([
        f"# {f['ticker']} research report",
        _section("1. Summary", txt["summary"]),
        _section("2. Data", data_table(f)),
        _section("3. Forecast", forecast_table(f),
                 img(figs, "price_fan", "price fan chart"),
                 img(figs, "forecast", "forecast"),
                 img(figs, "band", "realised return vs forecast band"),
                 img(figs, "scatter", "Mincer-Zarnowitz"),
                 img(figs, "err_dist", "forecast error distribution"),
                 img(figs, "pit", "PIT histogram")),
        _section("4. Backtest and comparison",
                 metrics_table(f), verdicts,
                 img(figs, "rmse", "rmse"),
                 img(figs, "ratio", "rmse ratio vs naive"),
                 img(figs, "dm", "dm p-values"),
                 img(figs, "coverage", "coverage and width")),
        _section("5. Volatility", garch_table(f), img(figs, "garch", "garch"),
                 img(figs, "vol_hist", "conditional vs realised volatility"),
                 img(figs, "vol_real", "forecast vs realised volatility"),
                 img(figs, "garch_diag", "GARCH residual diagnostics"),
                 img(figs, "nic", "news impact curve")),
        _section("6. Interpretation", txt["interpretation"]),
        _section("7. Caveats", txt["caveats"], _bullets(_fixed_caveats(f))),
        _section("8. Appendix", appendix),
    ]) + "\n"
