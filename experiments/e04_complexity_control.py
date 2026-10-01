"""E04 (derived from E03): 1-SE complexity control, scaled-target MLP teacher, Pareto fronts. Original E03 docstring: real monthly series. Teacher (XGB, MLP) -> sparse symbolic surrogate (distilled / on labels / raw-space-augmented)
-> fused raw formula. Compared with seasonal-naive, Holt-Winters ETS. One-step and 24-step recursive forecasts."""
import sys, pathlib, json, warnings
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from mt.data import make_real_pipeline, window_xy, RAW_REAL
from mt.teachers import make_teacher
from mt.surrogates import sparse_select
from mt.fuse import fuse, to_callable, eval_on_raw
from mt.metrics import rmse, smape
from mt.transforms import n_nodes

D = ROOT / "data"; OUT = ROOT / "results" / "e04"; OUT.mkdir(parents=True, exist_ok=True)
def load():
    out = {}
    a = pd.read_csv(D / "airline-passengers.csv"); out["airline"] = (a.Passengers.values.astype(float), pd.to_datetime(a.Month).dt.month.values)
    c = pd.read_csv(D / "monthly-car-sales.csv"); out["carsales"] = (c.Sales.values.astype(float), pd.to_datetime(c.Month).dt.month.values)
    co = pd.read_csv(D / "co2_weekly.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna().resample("MS").mean().dropna()
    out["co2"] = (co.values.astype(float), co.index.month.values)
    s = pd.read_csv(D / "monthly-sunspots.csv"); out["sunspots"] = (s.Sunspots.values[-600:].astype(float), pd.to_datetime(s.Month).dt.month.values[-600:])
    return out

def recursive(predict_raw, y, months, start, H):
    hist = list(y[:start]); preds = []
    for h in range(H):
        t = start + h
        raw = np.array([[hist[t - 1], hist[t - 2], hist[t - 3], hist[t - 6], hist[t - 12], months[t]]])
        p = float(predict_raw(raw)[0]); p = min(max(p if np.isfinite(p) else 0.0, 0.0), 10 * max(y[:start])); preds.append(p); hist.append(p)
    return np.array(preds)

rows = []; formulas = {}; pareto = []
H = 24
for name, (y, months) in load().items():
    Hn = H if name != "sunspots" else 60
    n = len(y); idx, X, T = window_xy(y, months)
    te = idx >= n - Hn; va = (idx >= n - 2 * Hn) & ~te; tr = idx < n - 2 * Hn
    pipe = make_real_pipeline().fit(X[tr]); k = len(pipe.names); S = [f"z{i}" for i in range(k)]
    zex = pipe.z_exprs()
    Z = {"tr": pipe.transform(X[tr]), "va": pipe.transform(X[va]), "te": pipe.transform(X[te])}
    yt = T[te]
    # --- baselines ---
    def add(method, complexity, p_one, p_rec):
        rows.append(dict(series=name, method=method, complexity=complexity,
                         one_rmse=rmse(yt, p_one) if p_one is not None else np.nan, one_smape=smape(yt, p_one) if p_one is not None else np.nan,
                         rec_rmse=rmse(yt, p_rec), rec_smape=smape(yt, p_rec)))
    sn_one = X[te][:, 4]
    add("seasonal_naive", 1, sn_one, recursive(lambda r: r[:, 4], y, months, n - Hn, Hn))
    try:
        ets = ExponentialSmoothing(y[: n - Hn], trend="add", seasonal="mul", seasonal_periods=12, damped_trend=True).fit()
        add("holt_winters", np.nan, None, ets.forecast(Hn))
    except Exception as e:
        print("ets fail", name, e)
    rng = np.random.default_rng(0)
    for tname in ["xgb", "mlp"]:
        t = make_teacher(tname, 0).fit(Z["tr"], T[tr])
        tpred = lambda r, t=t: t.predict(pipe.transform(r))
        add(f"teacher_{tname}", np.nan, tpred(X[te]), recursive(tpred, y, months, n - Hn, Hn))
        # raw-space augmentation: multiplicative jitter of each lag (2%), month kept; labels from teacher
        Xr = X[tr]; ii = rng.integers(0, len(Xr), 3000)
        Xa = Xr[ii].copy(); Xa[:, :5] *= np.exp(rng.normal(0, 0.03, (3000, 5)))
        Xq = {"labels": (X[tr], T[tr]), "distill": (X[tr], t.predict(Z["tr"])),
              }
        for qn, (Xs, ys) in Xq.items():
            if qn == "labels" and tname == "mlp": continue
            g, nt, path = sparse_select(pipe.transform(Xs), ys, Z["va"], T[va], rule="1se", return_path=True)
            F = fuse(g, zex); fF = to_callable(F, RAW_REAL)
            pr = lambda r, f=fF: eval_on_raw(f, r)
            m = f"sym[{qn}]" + ("" if qn == "labels" else f"<-{tname}")
            add(m, n_nodes(F), pr(X[te]), recursive(pr, y, months, n - Hn, Hn))
            if qn == "distill":
                for npar, eva, ex in path:
                    Fp = fuse(ex, zex); fp = to_callable(Fp, RAW_REAL); po = eval_on_raw(fp, X[te])
                    pareto.append(dict(series=name, teacher=tname, terms=npar, val_rmse=eva, fused_nodes=n_nodes(Fp), te_one_smape=smape(yt, po),
                                       te_rec_smape=smape(yt, recursive(lambda r, f=fp: eval_on_raw(f, r), y, months, n - Hn, Hn))))
            formulas[f"{name}|{m}"] = dict(g=str(g), g_nodes=n_nodes(g), fused_nodes=n_nodes(F))
    print(name, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "metrics.csv", index=False); pd.DataFrame(pareto).to_csv(OUT / "pareto.csv", index=False); json.dump(formulas, open(OUT / "formulas.json", "w"), indent=1)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(df.round(2).to_string())
