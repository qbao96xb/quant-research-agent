# src/report/tables.py
import math


# ---------- helpers ----------
def _fmt(v, spec=".4f", missing="-"):
    """Format a number safely. None, NaN, inf and non-numbers become `missing`."""
    if isinstance(v, bool) or v is None:
        return missing
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            return missing
        try:
            return format(v, spec)
        except (ValueError, TypeError):
            return str(v)
    return str(v)


def _cell(c):
    return str(c).replace("|", "\\|").replace("\n", " ")


def md_table(headers, rows):
    line = lambda r: "| " + " | ".join(_cell(c) for c in r) + " |"
    sep = "|" + "|".join(":--" if i == 0 else "--:" for i in range(len(headers))) + "|"
    rows = rows or [["-"] * len(headers)]
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


# ---------- tables ----------
def data_table(f):
    d = f.get("describe", {})
    n = d.get("n")
    n_txt = _fmt(int(n), "d") if isinstance(n, (int, float)) and math.isfinite(n) else "-"
    rows = [
        ["Start", _fmt(d.get("start"), "")],
        ["End", _fmt(d.get("end") or d.get("last_date"), "")],
        ["Observations (n)", n_txt],
        ["Mean daily log return", _fmt(d.get("mean_ret"), ".5f")],
        ["Daily volatility", _fmt(d.get("vol_daily"), ".5f")],
        ["Annualized vol (20d)", _fmt(d.get("vol_20d_ann"), ".3f")],
        ["Annualized vol (1y)", _fmt(d.get("vol_1y_ann"), ".3f")],
        ["Last close", _fmt(d.get("last_close"), ".2f")],
    ]
    return md_table(["Item", "Value"], rows)

def forecast_table(f):
    rows = [[m, _fmt(r.get("mean")), _fmt(r.get("lo")), _fmt(r.get("hi"))]
            for m, r in f.get("forecast", {}).items()]
    return md_table(["Model", "Mean", "Lo 90%", "Hi 90%"], rows)


def metrics_table(f):
    show_width = all("width" in r for r in f["metrics"].values())
    rows = []
    for m, r in f["metrics"].items():
        dm = f.get("dm", {}).get(m, {}).get("p_value")
        row = [m, _fmt(r.get("mae"), ".5f"), _fmt(r.get("rmse"), ".5f"),
               _fmt(r.get("diracc"), ".3f"), _fmt(r.get("cov90"), ".3f")]
        if show_width:
            row.append(_fmt(r.get("width"), ".4f"))
        row.append(_fmt(dm, ".3f"))
        rows.append(row)
    headers = ["Model", "MAE", "RMSE", "DirAcc", "Cov90"]
    if show_width:
        headers.append("Width")
    headers.append("DM p-value")
    return md_table(headers, rows)


def garch_table(f):
    g = f.get("garch", {})
    rows = [[k, _fmt(v, ".5f")] for k, v in g.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return md_table(["Item", "Value"], rows)
