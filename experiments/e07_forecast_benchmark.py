"""E07: 7 real monthly series, rolling-origin-CV 1-SE complexity control, library ablation, ETS/seasonal-naive/XGB baselines.
Test: one-step over last 24 months + 12-step recursive from origins n-24 and n-12 (averaged)."""
import sys, pathlib, json, warnings
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from mt.data import make_real_pipeline_kind, window_xy, RAW_REAL
from mt.teachers import make_teacher
from mt.surrogates import sparse_cv_select
from mt.fuse import fuse, to_callable, eval_on_raw
from mt.metrics import smape
from mt.transforms import n_nodes
D = ROOT / "data"; OUT = ROOT / "results" / "e07"; OUT.mkdir(parents=True, exist_ok=True)
def rd(f, col, m_fn=None):
    d = pd.read_csv(D / f); m = pd.to_datetime(d.Month).dt.month.values if m_fn is None else m_fn(d.Month); return d[col].values.astype(float), m
S = {"airline": rd("airline-passengers.csv", "Passengers"), "carsales": rd("monthly-car-sales.csv", "Sales"),
     "robberies": rd("monthly-robberies.csv", "Robberies"), "meantemp": rd("monthly-mean-temp.csv", "Temperature"),
     "writingpaper": rd("monthly-writing-paper-sales.csv", "Sales", lambda s: s.str.split("-").str[1].astype(int).values)}
co = pd.read_csv(D / "co2_weekly.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna().resample("MS").mean().dropna(); S["co2"] = (co.values.astype(float), co.index.month.values)
sn = pd.read_csv(D / "monthly-sunspots.csv"); S["sunspots"] = (sn.Sunspots.values[-600:].astype(float), pd.to_datetime(sn.Month).dt.month.values[-600:])

def rec(pred, y, months, start, H):
    hist = list(y[:start]); out = []
    for h in range(H):
        t = start + h; raw = np.array([[hist[t - 1], hist[t - 2], hist[t - 3], hist[t - 6], hist[t - 12], months[t]]])
        p = float(pred(raw)[0]); p = min(max(p if np.isfinite(p) else 0.0, 0.0), 10 * max(y[:start])); out.append(p); hist.append(p)
    return np.array(out)
rows = []; forms = {}
for name, (y, mo) in S.items():
    n = len(y); idx, X, T = window_xy(y, mo); Hn = 24; te = idx >= n - Hn; tr = idx < n - Hn; ntr = tr.sum(); yt = T[te]
    def evalm(method, cx, pred):  # pred: raw matrix -> predictions
        one = smape(yt, pred(X[te])); r = np.mean([smape(y[s:s + 12], rec(pred, y, mo, s, 12)) for s in (n - 24, n - 12)])
        rows.append(dict(series=name, method=method, nodes=cx, one_smape=one, rec12_smape=r))
    evalm("seasonal_naive", 1, lambda r: r[:, 4])
    try:
        sm = []
        for s in (n - 24, n - 12):
            f = ExponentialSmoothing(y[:s], trend="add", seasonal="mul" if y.min() > 0 else "add", seasonal_periods=12, damped_trend=True).fit().forecast(12); sm.append(smape(y[s:s + 12], f))
        rows.append(dict(series=name, method="holt_winters", nodes=np.nan, one_smape=np.nan, rec12_smape=float(np.mean(sm))))
    except Exception as e: print("ets", name, e)
    for kind in ["full", "nolog", "nofourier"]:
        pipe = make_real_pipeline_kind(kind).fit(X[tr]); Ztr = pipe.transform(X[tr]); k = Ztr.shape[1]; zex = pipe.z_exprs()
        folds = [ntr - 36, ntr - 24, ntr - 12]
        if kind == "full":
            xg = make_teacher("xgb", 0).fit(Ztr, T[tr]); evalm("teacher_xgb", np.nan, lambda r: xg.predict(pipe.transform(r)))
        for qn, tgt in ([("labels", T[tr]), ("distill", xg.predict(Ztr))] if kind == "full" else [("labels", T[tr])]):
            if qn == "distill" and kind != "full": continue
            g, nt = sparse_cv_select(Ztr, tgt, T[tr], folds)
            F = fuse(g, zex); fF = to_callable(F, RAW_REAL); evalm(f"sym[{qn}]/{kind}", n_nodes(F), lambda r, f=fF: eval_on_raw(f, r))
            forms[f"{name}|{qn}|{kind}"] = dict(terms=nt, fused_nodes=n_nodes(F), expr=str(sp.N(F, 4)) if n_nodes(F) < 120 else "(long)")
    print(name, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "metrics.csv", index=False); json.dump(forms, open(OUT / "formulas.json", "w"), indent=1)
pd.set_option("display.width", 200)
for c in ["one_smape", "rec12_smape", "nodes"]:
    print("\n==", c); print(df.pivot(index="method", columns="series", values=c).round(2).to_string())
