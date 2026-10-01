"""Verification benchmark: 3-5 datasets per PROBLEM TYPE. run(extractor_cls, trial) -> trials/LEADERBOARD_generalize.csv + outputs/<trial>_generalize.txt"""
import sys, pathlib, warnings, time, io; ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "trials")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.metrics import r2_score, roc_auc_score
import xgboost as xgb
from bench import mlp, T_
D = ROOT / "data"
def ident(names): return {n: (lambda e, L, n=n: e[n]) for n in names}
def onehot(col, codes):  # exact indicator for integer codes (Lagrange polynomial; valid in numpy and sympy)
    def mk(c):
        def f(e, L):
            v = 1
            for o in codes:
                if o != c: v = v * (e[col] - o) / (c - o)
            return v
        return f
    return {f"{col}_{c}": mk(c) for c in codes[1:]}

def tasks():
    rng = np.random.default_rng(0); out = []
    # TYPE physical_law (4): MLP teacher on 2% noisy data, truth known
    laws = {"gravity": (lambda a, b, c: a * b / c**2, ["m1", "m2", "r"]), "pendulum": (lambda L_, g: 2 * np.pi * np.sqrt(L_ / g), ["L", "g"]),
            "lens": (lambda u, v: u * v / (u + v), ["u", "v"]), "rc_decay": (lambda V, t, R: V * np.exp(-t / R), ["V0", "t", "R"])}
    for nm, (f, cols) in laws.items():
        k = len(cols); X = rng.uniform(1, 5, (1000, k)); y = f(*X.T); yn = y + rng.normal(0, .02 * y.std(), 1000); Xte = rng.uniform(1, 5, (2000, k)); Xex = rng.uniform(1, 7, (2000, k))
        out.append(dict(type="physical_law", name=nm, model=mlp().fit(X, yn), pre=ident(cols), raw=cols, model_input="raw", logit=False, Xtr=X, Xte=Xte, yte=f(*Xte.T), kind="reg", Xex=Xex, yex=f(*Xex.T)))
    # TYPE tabular_regression (3): MLP teacher
    d = load_diabetes(scaled=False); nm = [n.replace(" ", "") for n in d.feature_names]; idx = rng.permutation(442)
    out.append(dict(type="tabular_regression", name="diabetes", model=mlp((32,), 1.0).fit(d.data[idx[:300]], d.target[idx[:300]]), pre=ident(nm), raw=nm, model_input="raw", logit=False, Xtr=d.data[idx[:300]], Xte=d.data[idx[300:]], yte=d.target[idx[300:]], kind="reg"))
    b = pd.read_csv(D / "boston.csv"); cols = ["crim", "rm", "age", "dis", "tax", "ptratio", "lstat", "nox"]; X = b[cols].values; y = b.medv.values; idx = rng.permutation(len(X)); tr, te = idx[:350], idx[350:]
    pre = {"log_crim": lambda e, L: L.log(1 + e["crim"]), "rm": lambda e, L: e["rm"], "age": lambda e, L: e["age"], "log_dis": lambda e, L: L.log(e["dis"]), "tax": lambda e, L: e["tax"], "ptratio": lambda e, L: e["ptratio"], "log_lstat": lambda e, L: L.log(e["lstat"]), "nox": lambda e, L: e["nox"]}
    out.append(dict(type="tabular_regression", name="boston", model=mlp((32,), 1.0).fit(T_(pre, cols, X[tr]), y[tr]), pre=pre, raw=cols, model_input="features", logit=False, Xtr=X[tr], Xte=X[te], yte=y[te], kind="reg"))
    a = pd.read_csv(D / "mpg.csv").dropna(subset=["horsepower"]); a["origin"] = a.origin.map({"usa": 1, "europe": 2, "japan": 3})
    cols = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]; X = a[cols].values.astype(float); y = a.mpg.values; idx = rng.permutation(len(X)); tr, te = idx[:280], idx[280:]
    pre = {"log_weight": lambda e, L: L.log(e["weight"]), "log_hp": lambda e, L: L.log(e["horsepower"]), "displacement": lambda e, L: e["displacement"], "acceleration": lambda e, L: e["acceleration"], "model_year": lambda e, L: e["model_year"], "cylinders": lambda e, L: e["cylinders"], **onehot("origin", [1, 2, 3])}
    out.append(dict(type="tabular_regression", name="auto_mpg", model=mlp((32,), 1.0).fit(T_(pre, cols, X[tr]), y[tr]), pre=pre, raw=cols, model_input="features", logit=False, Xtr=X[tr], Xte=X[te], yte=y[te], kind="reg"))
    # TYPE forecasting (5): XGB teacher on engineered features, raw = lags + month
    def rd(f, col):
        x = pd.read_csv(D / f); return x[col].values.astype(float), pd.to_datetime(x.Month).dt.month.values
    series = {"airline": rd("airline-passengers.csv", "Passengers"), "carsales": rd("monthly-car-sales.csv", "Sales"), "robberies": rd("monthly-robberies.csv", "Robberies"), "meantemp": rd("monthly-mean-temp.csv", "Temperature")}
    co = pd.read_csv(D / "co2_weekly.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna().resample("MS").mean().dropna(); series["co2"] = (co.values, co.index.month.values)
    for nm, (s, mo) in series.items():
        t = np.arange(12, len(s)); Xr = np.column_stack([s[t-1], s[t-2], s[t-12], mo[t]]); yr = s[t]; n = len(t) - 24; raw = ["l1", "l2", "l12", "month"]
        pre = {"l1": lambda e, L: e["l1"], "l2": lambda e, L: e["l2"], "l12": lambda e, L: e["l12"], "log_l1": lambda e, L: L.log(e["l1"]), "log_l12": lambda e, L: L.log(e["l12"]),
               "sin_m": lambda e, L: L.sin(2 * L.pi * e["month"] / 12), "cos_m": lambda e, L: L.cos(2 * L.pi * e["month"] / 12)}
        m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05).fit(T_(pre, raw, Xr[:n]), yr[:n])
        out.append(dict(type="forecasting", name=nm, model=m, pre=pre, raw=raw, model_input="features", logit=False, Xtr=Xr[:n], Xte=Xr[n:], yte=yr[n:], kind="reg"))
    # TYPE classification (4): MLP classifier, probability
    def clf(): return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=3000, random_state=0))
    d = pd.read_csv(D / "titanic.csv"); d["sex"] = (d.sex == "female").astype(float); d["age"] = d.age.fillna(d.age.median())
    cols = ["pclass", "sex", "age", "sibsp", "fare"]; X = d[cols].values.astype(float); y = d.survived.values; idx = rng.permutation(len(X)); tr, te = idx[:600], idx[600:]
    pre = {"pclass": lambda e, L: e["pclass"], "sex": lambda e, L: e["sex"], "age": lambda e, L: e["age"], "sibsp": lambda e, L: e["sibsp"], "log_fare": lambda e, L: L.log(1 + e["fare"])}
    out.append(dict(type="classification", name="titanic", model=clf().fit(T_(pre, cols, X[tr]), y[tr]), pre=pre, raw=cols, model_input="features", logit=True, Xtr=X[tr], Xte=X[te], yte=y[te], kind="cls"))
    p = pd.read_csv(D / "pima.csv"); cols = list(p.columns[:-1]); X = p[cols].values.astype(float); y = p.Outcome.values; idx = rng.permutation(len(X)); tr, te = idx[:540], idx[540:]
    out.append(dict(type="classification", name="pima_diabetes", model=clf().fit(X[tr], y[tr]), pre=ident(cols), raw=cols, model_input="features", logit=True, Xtr=X[tr], Xte=X[te], yte=y[te], kind="cls"))
    bc = load_breast_cancer(); keep = [0, 1, 4, 6, 7, 20, 21, 27]; cols = [bc.feature_names[i].replace(" ", "_") for i in keep]; X = bc.data[:, keep]; y = bc.target; idx = rng.permutation(len(X)); tr, te = idx[:400], idx[400:]
    out.append(dict(type="classification", name="breast_cancer", model=clf().fit(X[tr], y[tr]), pre=ident(cols), raw=cols, model_input="features", logit=True, Xtr=X[tr], Xte=X[te], yte=y[te], kind="cls"))
    pg = pd.read_csv(D / "penguins.csv").dropna(); cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]; X = pg[cols].values; y = (pg.species == "Chinstrap").values.astype(int); idx = rng.permutation(len(X)); tr, te = idx[:230], idx[230:]
    out.append(dict(type="classification", name="penguins", model=clf().fit(X[tr], y[tr]), pre=ident(cols), raw=cols, model_input="features", logit=True, Xtr=X[tr], Xte=X[te], yte=y[te], kind="cls"))
    return out

def run(extractor_cls, trial, only_types=None):
    rows = []; buf = io.StringIO(); lb = ROOT / "trials/LEADERBOARD_generalize.csv"
    for tk in tasks():
        if only_types and tk["type"] not in only_types: continue
        t0 = time.time()
        try: ex = extractor_cls(tk["model"], tk["pre"], tk["raw"], model_input=tk["model_input"], logit=tk["logit"]).fit(tk["Xtr"])
        except Exception as err: print("FAIL", trial, tk["name"], repr(err)[:200], flush=True); continue
        pf, pm = np.nan_to_num(ex.predict(tk["Xte"])), ex.model_predict(tk["Xte"]); sc = r2_score if tk["kind"] == "reg" else roc_auc_score
        r = dict(trial=trial, type=tk["type"], task=tk["name"], model_score=sc(tk["yte"], pm), formula_score=sc(tk["yte"], pf), fidelity=r2_score(pm, pf),
                 nodes=sum(1 for _ in sp.preorder_traversal(ex.full_formula)), secs=round(time.time() - t0, 1), formula=str(sp.N(ex.full_formula, 5)))
        if "Xex" in tk: r["extrap_r2"] = r2_score(tk["yex"], np.nan_to_num(ex.predict(tk["Xex"])))
        rows.append(r); pd.DataFrame([r]).to_csv(lb, mode="a", header=not lb.exists(), index=False)
        line = f"[{trial}] {tk['type']:18s} {tk['name']:14s} model={r['model_score']:.3f} formula={r['formula_score']:.3f} fid={r['fidelity']:.3f} nodes={r['nodes']} " + (f"extrap={r['extrap_r2']:.3f}" if 'extrap_r2' in r else "") + f"\n    y = {r['formula']}"
        print(line, flush=True); buf.write(line + "\n")
    (ROOT / "trials/outputs" / f"{trial}_generalize.txt").write_text(buf.getvalue()); return pd.DataFrame(rows)
