"""Run:  python demo.py      (about 3-5 minutes)
One example per problem type: train a model, extract its formula, predict from RAW data with the formula only, export it."""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, roc_auc_score
import xgboost as xgb
from model2formula import extract
HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE / "output"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
def mlp(h=(64, 64), a=1e-3): return TransformedTargetRegressor(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=h, activation="tanh", alpha=a, max_iter=4000, random_state=0)), transformer=StandardScaler())
def feats(pre, raw, X): return np.column_stack([np.asarray(f({n: X[:, i] for i, n in enumerate(raw)}, np), float) * np.ones(len(X)) for f in pre.values()])
def show(name, res, Xte, yte, metric):
    pf, pm = res.predict(Xte), res.model_predict(Xte)
    print(f"\n===== {name}  [method: {res.method}]\nformula on raw inputs:\n  y = {sp.N(res.full_formula, 4)}")
    print(f"test {metric.__name__}: model={metric(yte, pm):.3f}  formula={metric(yte, pf):.3f}  | formula-vs-model R2={r2_score(pm, pf):.3f}")
    res.export(OUT / f"{name}.py", name); print(f"exported -> output/{name}.py")

# 1) PHYSICAL LAW: neural net learns gravity from noisy data
X = rng.uniform(1, 5, (1000, 3)); y = X[:, 0] * X[:, 1] / X[:, 2] ** 2; Xte = rng.uniform(1, 5, (1000, 3))
model = mlp().fit(X, y + rng.normal(0, .02 * y.std(), 1000))
pre = {k: (lambda e, L, k=k: e[k]) for k in ("m1", "m2", "r")}
show("gravity", extract(model, pre, ["m1", "m2", "r"], X, "physical_law", model_input="raw"), Xte, Xte[:, 0] * Xte[:, 1] / Xte[:, 2] ** 2, r2_score)

# 2) FORECASTING: XGBoost on engineered features, formula takes raw lags + month
a = pd.read_csv(HERE / "data/airline-passengers.csv"); s = a.Passengers.values.astype(float); mo = pd.to_datetime(a.Month).dt.month.values; t = np.arange(12, len(s))
raw = ["l1", "l2", "l12", "month"]; Xr = np.column_stack([s[t-1], s[t-2], s[t-12], mo[t]]); yr = s[t]; n = len(t) - 24
pre = {"l1": lambda e, L: e["l1"], "l2": lambda e, L: e["l2"], "l12": lambda e, L: e["l12"], "log_l1": lambda e, L: L.log(e["l1"]), "log_l12": lambda e, L: L.log(e["l12"]),
       "sin_m": lambda e, L: L.sin(2 * L.pi * e["month"] / 12), "cos_m": lambda e, L: L.cos(2 * L.pi * e["month"] / 12)}
model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05).fit(feats(pre, raw, Xr[:n]), yr[:n])
show("airline_forecast", extract(model, pre, raw, Xr[:n], "forecasting"), Xr[n:], yr[n:], r2_score)

# 3) TABULAR REGRESSION: neural net on diabetes
d = load_diabetes(scaled=False); nm = [c.replace(" ", "") for c in d.feature_names]; idx = rng.permutation(442); tr, te = idx[:300], idx[300:]
model = mlp((32,), 1.0).fit(d.data[tr], d.target[tr])
show("diabetes", extract(model, {c: (lambda e, L, c=c: e[c]) for c in nm}, nm, d.data[tr], "tabular_regression", model_input="raw"), d.data[te], d.target[te], r2_score)

# 4) CLASSIFICATION: neural net on titanic (preprocessing: log(1+fare)); formula gives survival probability
tt = pd.read_csv(HERE / "data/titanic.csv"); tt["sex"] = (tt.sex == "female").astype(float); tt["age"] = tt.age.fillna(tt.age.median())
cols = ["pclass", "sex", "age", "sibsp", "fare"]; X = tt[cols].values.astype(float); y = tt.survived.values; idx = rng.permutation(len(X)); tr, te = idx[:600], idx[600:]
pre = {"pclass": lambda e, L: e["pclass"], "sex": lambda e, L: e["sex"], "age": lambda e, L: e["age"], "sibsp": lambda e, L: e["sibsp"], "log_fare": lambda e, L: L.log(1 + e["fare"])}
model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=3000, random_state=0)).fit(feats(pre, cols, X[tr]), y[tr])
show("titanic_probability", extract(model, pre, cols, X[tr], "classification", classification=True), X[te], y[te], roc_auc_score)

# using an exported formula file: no model, only numpy
import importlib.util; spec = importlib.util.spec_from_file_location("f", OUT / "gravity.py"); f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
print("\nexported gravity.py: predict(m1=2, m2=3, r=1.5) =", f.predict(m1=2.0, m2=3.0, r=1.5), " (truth", 2 * 3 / 1.5 ** 2, ")  in_domain:", f.in_domain(m1=2.0, m2=3.0, r=1.5))
