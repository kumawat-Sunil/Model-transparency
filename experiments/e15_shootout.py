"""E15 SHOOT-OUT: which approach turns a trained model into a usable expression?
Approaches: operon<-teacher, operon<-labels, sparse-poly<-teacher, gplearn<-teacher, linear baseline, small tree (depth 3).
Tasks: 6 known laws (MLP teacher, n=500, 5% noise), diabetes (real), airline (real, forecasting).
Scores: test R2 (vs noise-free truth for laws), fidelity R2 vs teacher, extrapolation R2 (laws, 1.5x box), nodes."""
import sys, pathlib, warnings, time; ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp
from pyoperon.sklearn import SymbolicRegressor as Operon
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import xgboost as xgb
from mt.surrogates import gp_fit, gp_to_sympy
OPS = "add,sub,mul,div,constant,variable,sin,cos,log,exp,sqrt,square"
def nn(): return TransformedTargetRegressor(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", alpha=1e-3, max_iter=4000, random_state=0)), transformer=StandardScaler())
def count(e): return sum(1 for _ in sp.preorder_traversal(e))

def m_operon(X, y, L=20):
    m = Operon(allowed_symbols=OPS, generations=200, population_size=1000, max_length=L, n_threads=4, random_state=0).fit(X, y)
    loc = {f"X{i+1}": sp.Symbol(f"x{i}") for i in range(X.shape[1])}; loc["square"] = lambda a: a**2
    e = sp.sympify(m.get_model_string(m.model_, precision=6).replace("^", "**"), locals=loc); return m.predict, count(e), e
def m_sparse(X, y):
    pf = PolynomialFeatures(2, include_bias=False); sc = StandardScaler(); A = sc.fit_transform(pf.fit_transform(X)); las = LassoCV(cv=5, random_state=0).fit(A, y)
    return (lambda Z: las.predict(sc.transform(pf.transform(Z)))), int(3 * (las.coef_ != 0).sum() + 1), None
def m_gplearn(X, y):
    sub = np.random.default_rng(0).choice(len(X), min(1500, len(X)), replace=False); sr = gp_fit(X[sub], y[sub], 2e-3, pop=1000, gens=20); g, _ = gp_to_sympy(sr, X.shape[1]); return sr.predict, count(g), g
def m_linear(X, y): lr = LinearRegression().fit(X, y); return lr.predict, 2 * X.shape[1] + 1, None
def m_tree(X, y): t = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y); return t.predict, t.tree_.node_count, None

LAWS = {"newton": (lambda a, b, c: a * b / c**2, 3), "kinetic": (lambda m, v: 0.5 * m * v**2, 2), "pendulum": (lambda L, g: 2 * np.pi * np.sqrt(L / g), 2),
        "lens": (lambda u, v: u * v / (u + v), 2), "rc_decay": (lambda V, t, R: V * np.exp(-t / R), 3), "range": (lambda v, th, g: v**2 * np.sin(2 * th) / g, 3)}
rows = []
def run(task, Xtr, ytr, teacher, Xte, yte_true, Xex=None, yex_true=None):
    tp = teacher.predict(Xte); rows.append(dict(task=task, approach="TEACHER", r2=r2_score(yte_true, tp), fid=1.0, ex_r2=r2_score(yex_true, teacher.predict(Xex)) if Xex is not None else np.nan, nodes=np.nan))
    yq = teacher.predict(Xtr)
    for name, fn, tgt in [("operon<-teacher", m_operon, yq), ("operon<-labels", m_operon, ytr), ("sparse_poly<-teacher", m_sparse, yq), ("gplearn<-teacher", m_gplearn, yq), ("linear<-labels", m_linear, ytr), ("tree_d3<-teacher", m_tree, yq)]:
        t0 = time.time(); pred, nodes, e = fn(Xtr, tgt); p = pred(Xte)
        rows.append(dict(task=task, approach=name, r2=r2_score(yte_true, p), fid=r2_score(tp, p), ex_r2=r2_score(yex_true, pred(Xex)) if Xex is not None else np.nan, nodes=nodes, secs=round(time.time() - t0, 1), expr=str(sp.N(e, 4)) if e is not None and nodes < 60 else ""))
    print(task, "done", flush=True)

for law, (f, k) in LAWS.items():
    rng = np.random.default_rng(0); box = lambda n, s=1: rng.uniform(1, 1 + 4 * s, (n, k))
    X = box(500); yt = f(*X.T); y = yt + rng.normal(0, 0.05 * yt.std(), len(yt)); Xte, Xex = box(2000), box(2000, 1.5)
    if law == "range": X[:, 1] = rng.uniform(0.1, 1.4, 500); Xte[:, 1] = rng.uniform(0.1, 1.4, 2000); Xex[:, 1] = rng.uniform(0.1, 1.4, 2000)
    if law == "range": y = f(*X.T) + rng.normal(0, 0.05 * f(*X.T).std(), 500)
    run(f"law:{law}", X, y, nn().fit(X, y), Xte, f(*Xte.T), Xex, f(*Xex.T))
d = load_diabetes(scaled=False); idx = np.random.default_rng(0).permutation(442); tr, te = idx[:300], idx[300:]
run("real:diabetes", d.data[tr], d.target[tr], nn().fit(d.data[tr], d.target[tr]), d.data[te], d.target[te])
a = pd.read_csv(ROOT / "data/airline-passengers.csv"); s = a.Passengers.values.astype(float); mo = pd.to_datetime(a.Month).dt.month.values; t = np.arange(12, len(s))
F = np.column_stack([s[t-1], s[t-3], s[t-12], np.log(s[t-1]), np.log(s[t-12]), np.sin(2*np.pi*mo[t]/12), np.cos(2*np.pi*mo[t]/12)]); n = len(t) - 24
run("real:airline", F[:n], s[t][:n], xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05).fit(F[:n], s[t][:n]), F[n:], s[t][n:])
df = pd.DataFrame(rows); (ROOT / "results/e15").mkdir(exist_ok=True); df.to_csv(ROOT / "results/e15/metrics.csv", index=False)
pd.set_option("display.width", 220)
for c in ("r2", "ex_r2", "fid", "nodes"): print(f"\n== {c}"); print(df.pivot(index="task", columns="approach", values=c).round(3).to_string())
