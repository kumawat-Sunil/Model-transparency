"""Results: model -> formula -> raw data -> prediction, on 4 problems."""
import sys, pathlib, warnings; ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score
import xgboost as xgb
from mt.model2formula import Model2Formula
def mlp(h=(64, 64), alpha=1e-3): return TransformedTargetRegressor(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=h, activation="tanh", alpha=alpha, max_iter=4000, random_state=0)), transformer=StandardScaler())
def show(title, m2f, Xte, yte, kind="reg"):
    pf, pm = m2f.predict(Xte), m2f.model_predict(Xte)
    sc = (lambda y, p: r2_score(y, p)) if kind == "reg" else (lambda y, p: roc_auc_score(y, p))
    print(f"\n################ {title}")
    print("PREPROCESSING T:", {k: str(e) for k, e in zip(m2f.pre, m2f.T_sym())})
    print("search chosen (representation, max_len, held-out fidelity):", m2f.choice)
    print("FUNCTION g(features):  y =", sp.N(m2f.formula, 5))
    print("FULL FORMULA on RAW:   y =", sp.N(m2f.full_formula, 5))
    print(f"test score  model={sc(yte, pm):.4f}   formula={sc(yte, pf):.4f}   agreement formula-vs-model R2={r2_score(pm, pf):.4f}")
    print(pd.DataFrame({"raw_input": [np.round(r, 2).tolist() for r in Xte[:5]], "model": pm[:5].round(3), "formula": pf[:5].round(3), "truth": np.asarray(yte[:5]).round(3)}).to_string(index=False))

# 1) Physics: model learned newton's gravity from noisy data (truth known)
rng = np.random.default_rng(0); X = rng.uniform(1, 5, (1000, 3)); y = X[:, 0] * X[:, 1] / X[:, 2] ** 2; yn = y + rng.normal(0, .05 * y.std(), 1000)
pre = {"m1": lambda e, L: e["m1"], "m2": lambda e, L: e["m2"], "r": lambda e, L: e["r"]}
model = mlp().fit(X, yn); Xte = rng.uniform(1, 5, (2000, 3))
show("GRAVITY  (MLP)", Model2Formula(model, pre, ["m1", "m2", "r"], model_input="raw").fit(X), Xte, Xte[:, 0] * Xte[:, 1] / Xte[:, 2] ** 2)

# 2) Diabetes (real), MLP on raw columns
d = load_diabetes(scaled=False); nm = [n.replace(" ", "") for n in d.feature_names]; idx = rng.permutation(442); tr, te = idx[:300], idx[300:]
pre = {n: (lambda e, L, n=n: e[n]) for n in nm}
model = mlp((32,), 1.0).fit(d.data[tr], d.target[tr])
show("DIABETES (MLP)", Model2Formula(model, pre, nm, model_input="raw", max_length=20).fit(d.data[tr]), d.data[te], d.target[te])

# 3) Airline forecasting (real): XGB uses engineered features; formula must work from raw lags + month
a = pd.read_csv(ROOT / "data/airline-passengers.csv"); s = a.Passengers.values.astype(float); mo = pd.to_datetime(a.Month).dt.month.values; t = np.arange(12, len(s))
Xr = np.column_stack([s[t-1], s[t-2], s[t-12], mo[t]]); yr = s[t]; n = len(t) - 24
pre = {"l1": lambda e, L: e["l1"], "l12": lambda e, L: e["l12"], "log_l1": lambda e, L: L.log(e["l1"]), "log_l2": lambda e, L: L.log(e["l2"]), "log_l12": lambda e, L: L.log(e["l12"]),
       "sin_m": lambda e, L: L.sin(2 * L.pi * e["month"] / 12), "cos_m": lambda e, L: L.cos(2 * L.pi * e["month"] / 12)}
m2f = Model2Formula(None, pre, ["l1", "l2", "l12", "month"], max_length=15)
m2f.model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05).fit(m2f.T(Xr[:n]), yr[:n])
show("AIRLINE forecast (XGBoost)", m2f.fit(Xr[:n]), Xr[n:], yr[n:])

# 4) Titanic classification: MLP on preprocessed (impute+log fare+codes) -> probability formula
d = pd.read_csv(ROOT / "data/titanic.csv"); d["sex"] = (d.sex == "female").astype(float); d["age"] = d.age.fillna(d.age.median())
cols = ["pclass", "sex", "age", "sibsp", "fare"]; X = d[cols].values.astype(float); y = d.survived.values; idx = rng.permutation(len(X)); tr, te = idx[:600], idx[600:]
pre = {"pclass": lambda e, L: e["pclass"], "sex": lambda e, L: e["sex"], "age": lambda e, L: e["age"], "sibsp": lambda e, L: e["sibsp"], "log_fare": lambda e, L: L.log(1 + e["fare"])}
m2f = Model2Formula(None, pre, cols, max_length=20, logit=True)
m2f.model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=3000, random_state=0)).fit(m2f.T(X[tr]), y[tr])
show("TITANIC survival probability (MLP)", m2f.fit(X[tr]), X[te], y[te], kind="cls")
