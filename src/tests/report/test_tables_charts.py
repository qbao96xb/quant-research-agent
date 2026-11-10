import pytest
from src.report.tables import md_table, metrics_table
from src.report import charts

FACTS_CHARTS = ["forecast_interval", "metrics_bar", "rmse_ratio",
                "dm_pvalues", "coverage_width", "garch_chart"]

def test_md_table_shape():
    t = md_table(["A", "B"], [[1, 2]]).splitlines()
    assert t[0] == "| A | B |"
    assert t[1] == "|:--|--:|"
    assert t[2] == "| 1 | 2 |"

def test_metrics_table_missing_diracc(facts):
    m = next(iter(facts["metrics"]))
    facts["metrics"][m]["diracc"] = None
    row = [l for l in metrics_table(facts).splitlines() if l.startswith(f"| {m} ")][0]
    assert " - " in row

def test_metrics_table_decimals(facts):
    line = metrics_table(facts).splitlines()[2]
    assert len(line.split("|")[2].strip().split(".")[1]) == 5  # MAE

@pytest.mark.parametrize("name", FACTS_CHARTS)
def test_facts_chart_creates_png(name, facts, tmp_path):
    rel = getattr(charts, name)(facts, tmp_path)
    p = tmp_path / rel
    assert p.exists() and p.stat().st_size > 1000


def test_build_all_without_preds(facts, tmp_path):
    figs = charts.build_all(facts, tmp_path)
    assert figs  # inspect the return type if this fails

# ---------- professional charts (synthetic data) ----------
import numpy as np
import pandas as pd


@pytest.fixture
def rich(facts):
    rng = np.random.default_rng(0)
    n, h = 250, facts["h"]
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 60 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    z = rng.standard_t(6, 400) * 0.6
    facts["history"] = dict(dates=[str(d)[:10] for d in dates], close=close.tolist())
    facts["paths"] = {m: [dict(price=float(close[-1]), cum_log_return=0.001 * k, hi=0.01 * np.sqrt(k + 1),
                               lo=-0.01 * np.sqrt(k + 1)) for k in range(h)] for m in facts["metrics"]}
    facts["garch"]["path"] = [dict(daily_sigma=0.01, sigma=0.01 * np.sqrt(k + 1)) for k in range(h)]
    facts["garch_diag"] = dict(model="GARCH(1,1,0)-t", dates=[str(d)[:10] for d in pd.bdate_range(end=dates[-1], periods=400)],
                               ret=(z * 0.01).tolist(), cond_vol=[0.01] * 400, std_resid=z.tolist(),
                               params={"omega": 0.02, "alpha[1]": 0.08, "beta[1]": 0.9, "nu": 6.0},
                               nu=6.0, vol="GARCH", o=0)
    rows = []
    for m in facts["metrics"]:
        for i in range(80):
            last = 60.0; sig = 0.02; ret = rng.normal(0, sig)
            rows.append(dict(ticker="KO", model=m, origin=pd.Timestamp("2024-06-01") + pd.Timedelta(7 * i, unit='D'),
                             target_date=pd.Timestamp("2024-06-08") + pd.Timedelta(7 * i, unit='D'), h=h, last=last,
                             actual=last * np.exp(ret), pred=last * np.exp(0.0 if m == "naive" else 0.001),
                             lower=last * np.exp(-1.645 * sig), upper=last * np.exp(1.645 * sig)))
    return facts, pd.DataFrame(rows)


def test_new_charts_build(rich, tmp_path):
    f, preds = rich
    figs = charts.build_all(f, tmp_path, preds=preds)
    for k in ("price_fan", "vol_hist", "garch_diag", "nic", "band", "scatter",
              "err_dist", "pit", "vol_real"):
        v = figs[k]
        assert not str(v).startswith("MISSING"), (k, v)
        for p in (v if isinstance(v, list) else [v]):
            assert (tmp_path / p).stat().st_size > 1000
