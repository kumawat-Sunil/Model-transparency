"""Fixed benchmark used by EVERY trial (do not change task definitions -> comparability).
run(extractor_cls, trial_name, **kw) -> prints formulas, writes trials/outputs/<trial>.txt and appends trials/LEADERBOARD.csv"""
import sys, pathlib, warnings, time, io, contextlib, datetime; ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, roc_auc_score
import xgboost as xgb

def mlp(h=(64, 64), alpha=1e-3): return TransformedTargetRegressor(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=h, activation="tanh", alpha=alpha, max_iter=4000, random_state=0)), transformer=StandardScaler())
T_ = lambda pre, raw, X: np.column_stack([np.asarray(f({n: X[:, i] for i, n in enumerate(raw)}, np), float) * np.ones(len(X)) for f in pre.values()])

def tasks():
    out = []; rng = np.random.default_rng(0)
    # 1 gravity (truth known)
    X = rng.uniform(1, 5, (1000, 3)); y = X[:, 0] * X[:, 1] / X[:, 2] ** 2; yn = y + rng.normal(0, .05 * y.std(), 1000); Xte = rng.uniform(1, 5, (2000, 3)); Xex = rng.uniform(1, 7, (2000, 3))
    pre = {k: (lambda e, L, k=k: e[k]) for k in ("m1", "m2", "r")}
    out.append(dict(name="gravity", model=mlp().fit(X, yn), pre=pre, raw=["m1", "m2", "r"], model_input="raw", logit=False, Xtr=X, Xte=Xte, yte=Xte[:, 0] * Xte[:, 1] / Xte[:, 2] ** 2, kind="reg",
                    Xex=Xex, yex=Xex[:, 0] * Xex[:, 1] / Xex[:, 2] ** 2, truth="m1*m2/r**2"))
    # 2 diabetes
    d = load_diabetes(scaled=False); nm = [n.replace(" ", "") for n in d.feature_names]; idx = rng.permutation(442); tr, te = idx[:300], idx[300:]
    out.append(dict(name="diabetes", model=mlp((32,), 1.0).fit(d.data[tr], d.target[tr]), pre={n: (lambda e, L, n=n: e[n]) for n in nm}, raw=nm, model_input="raw", logit=False, Xtr=d.data[tr], Xte=d.data[te], yte=d.target[te], kind="reg"))
    # 3 airline (XGB on engineered features)
    a = pd.read_csv(ROOT / "data/airline-passengers.csv"); s = a.Passengers.values.astype(float); mo = pd.to_datetime(a.Month).dt.month.values; t = np.arange(12, len(s))
    Xr = np.column_stack([s[t-1], s[t-2], s[t-12], mo[t]]); yr = s[t]; n = len(t) - 24; raw = ["l1", "l2", "l12", "month"]
    pre = {"l1": lambda e, L: e["l1"], "l12": lambda e, L: e["l12"], "log_l1": lambda e, L: L.log(e["l1"]), "log_l2": lambda e, L: L.log(e["l2"]), "log_l12": lambda e, L: L.log(e["l12"]),
           "sin_m": lambda e, L: L.sin(2 * L.pi * e["month"] / 12), "cos_m": lambda e, L: L.cos(2 * L.pi * e["month"] / 12)}
    m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05).fit(T_(pre, raw, Xr[:n]), yr[:n])
    out.append(dict(name="airline", model=m, pre=pre, raw=raw, model_input="features", logit=False, Xtr=Xr[:n], Xte=Xr[n:], yte=yr[n:], kind="reg"))
    # 4 titanic (MLP classifier, probability)
    d = pd.read_csv(ROOT / "data/titanic.csv"); d["sex"] = (d.sex == "female").astype(float); d["age"] = d.age.fillna(d.age.median())
    cols = ["pclass", "sex", "age", "sibsp", "fare"]; X = d[cols].values.astype(float); y = d.survived.values; idx = rng.permutation(len(X)); tr, te = idx[:600], idx[600:]
    pre = {"pclass": lambda e, L: e["pclass"], "sex": lambda e, L: e["sex"], "age": lambda e, L: e["age"], "sibsp": lambda e, L: e["sibsp"], "log_fare": lambda e, L: L.log(1 + e["fare"])}
    m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=3000, random_state=0)).fit(T_(pre, cols, X[tr]), y[tr])
    out.append(dict(name="titanic", model=m, pre=pre, raw=cols, model_input="features", logit=True, Xtr=X[tr], Xte=X[te], yte=y[te], kind="cls"))
    return out

def run(extractor_cls, trial, note="", **kw):
    buf = io.StringIO(); rows = []
    for tk in tasks():
        t0 = time.time()
        ex = extractor_cls(tk["model"], tk["pre"], tk["raw"], model_input=tk["model_input"], logit=tk["logit"], **kw).fit(tk["Xtr"])
        secs = time.time() - t0; pf, pm = ex.predict(tk["Xte"]), ex.model_predict(tk["Xte"])
        sc = r2_score if tk["kind"] == "reg" else roc_auc_score
        nodes = sum(1 for _ in sp.preorder_traversal(ex.full_formula))
        r = dict(trial=trial, task=tk["name"], model_score=sc(tk["yte"], pm), formula_score=sc(tk["yte"], pf), fidelity=r2_score(pm, pf), nodes=nodes, secs=round(secs, 1))
        if "Xex" in tk: r["extrap_r2"] = r2_score(tk["yex"], ex.predict(tk["Xex"]))
        rows.append(r)
        txt = (f"\n#### {trial} | {tk['name']}\nPREPROCESSING: { {k: str(e) for k, e in zip(tk['pre'], ex.T_sym())} }\n"
               f"choice: {getattr(ex, 'choice', '')}\nFORMULA (raw inputs): y = {sp.N(ex.full_formula, 5)}\n"
               + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items() if k not in ('trial', 'task')) + (f"\ntruth: {tk['truth']}" if "truth" in tk else ""))
        print(txt, flush=True); buf.write(txt + "\n")
    (ROOT / "trials/outputs" / f"{trial}.txt").write_text(f"{trial}  {datetime.datetime.now()}  {note}\n" + buf.getvalue())
    lb = ROOT / "trials/LEADERBOARD.csv"; df = pd.DataFrame(rows); df["note"] = note
    df.to_csv(lb, mode="a", header=not lb.exists(), index=False); return df
