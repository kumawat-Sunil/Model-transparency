"""E14 END-TO-END with Operon. Part 1: train NN/XGB, extract expression (Operon on teacher outputs). Part 2: raw data -> expression only (no model)."""
import sys, pathlib, warnings, re; ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp
from pyoperon.sklearn import SymbolicRegressor
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, roc_auc_score
import xgboost as xgb
from mt.metrics import smape
OPS = "add,sub,mul,div,constant,variable,sin,cos,log,exp,sqrt,square"

def operon(F, y, max_len, seed=0, gens=150):
    m = SymbolicRegressor(allowed_symbols=OPS, generations=gens, population_size=1000, max_length=max_len, n_threads=4, random_state=seed, optimizer_iterations=10).fit(F, y)
    s = m.get_model_string(m.model_, precision=6); loc = {f"X{i+1}": sp.Symbol(f"f{i}") for i in range(F.shape[1])}
    loc.update(dict(square=lambda a: a**2, aq=lambda a, b: a / sp.sqrt(1 + b**2)))
    return sp.sympify(s.replace("^", "**"), locals=loc)

def raw_eval(expr, names, X):  # PART 2: expression on raw columns only
    f = sp.lambdify(sp.symbols([f"f{i}" for i in range(len(names))]), expr, "numpy"); return np.asarray(f(*X.T), float) * np.ones(len(X))

out = []
def report(title, names, expr_by_len, truth_score, teacher_pred, Xte, yte, kind):
    print(f"\n=========== {title} ==========="); 
    for L, e in expr_by_len.items():
        p = raw_eval(e, names, Xte); sc = r2_score(yte, p) if kind == "reg" else roc_auc_score(yte, p)
        fid = r2_score(teacher_pred, p) if kind == "reg" else np.corrcoef(teacher_pred, p)[0, 1]
        shown = e.subs({sp.Symbol(f"f{i}"): sp.Symbol(n) for i, n in enumerate(names)})
        print(f"[max_len={L}] nodes={sum(1 for _ in sp.preorder_traversal(e))}  test {'R2' if kind=='reg' else 'AUC'}={sc:.3f}  fidelity_to_teacher={fid:.3f}\n   y = {sp.N(shown, 5)}"); out.append((title, L, sc, fid))
    print(f"   teacher/baseline score: {truth_score:.3f}")

# ---------- 1. Diabetes (real, raw units, 10 inputs) ----------
d = load_diabetes(scaled=False); X, y = d.data, d.target; names = list(d.feature_names)
rng = np.random.default_rng(0); idx = rng.permutation(len(X)); tr, te = idx[:300], idx[300:]
nn = TransformedTargetRegressor(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,), alpha=1.0, max_iter=3000, random_state=0)), transformer=StandardScaler()).fit(X[tr], y[tr])
tp = nn.predict(X[te]); Xq = X[tr]; yq = nn.predict(Xq)
report("DIABETES: MLP -> expression on raw inputs", names, {L: operon(Xq, yq, L) for L in (7, 15, 30)}, r2_score(y[te], tp), tp, X[te], y[te], "reg")

# ---------- 2. Airline forecasting (real), XGB teacher ----------
a = pd.read_csv(ROOT / "data/airline-passengers.csv"); s = a.Passengers.values.astype(float); mo = pd.to_datetime(a.Month).dt.month.values
t = np.arange(12, len(s)); Xr = np.column_stack([s[t-1], s[t-2], s[t-3], s[t-12], mo[t]]); yr = s[t]; nm = ["l1", "l2", "l3", "l12", "month"]
ntr = len(t) - 24
feat = lambda X: np.column_stack([X[:, 0], X[:, 1], X[:, 2], X[:, 3], np.log(X[:, 0]), np.log(X[:, 3]), np.sin(2*np.pi*X[:, 4]/12), np.cos(2*np.pi*X[:, 4]/12)])
fn = ["l1", "l2", "l3", "l12", "log_l1", "log_l12", "sin_m", "cos_m"]
tm = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05).fit(Xr[:ntr], yr[:ntr]); tp = tm.predict(Xr[ntr:])
Fq = feat(Xr[:ntr]); yq = tm.predict(Xr[:ntr])
def fused_eval(e, X): return raw_eval(e, fn, feat(X))  # features are explicit functions of raw lags/month (part of T)
for lab, tgt in (("distilled from XGB", yq), ("fit on labels", yr[:ntr])):
    print(f"\n=========== AIRLINE ({lab}) -> expression of raw lags ==========="); 
    for L in (7, 15, 30):
        e = operon(Fq, tgt, L); p = fused_eval(e, Xr[ntr:]); shown = e.subs({sp.Symbol(f"f{i}"): sp.Symbol(n) for i, n in enumerate(fn)})
        print(f"[max_len={L}] test sMAPE one-step={smape(yr[ntr:], p):.2f}%  (XGB teacher {smape(yr[ntr:], tp):.2f}%, seasonal-naive {smape(yr[ntr:], Xr[ntr:,3]):.2f}%)\n   y = {sp.N(shown,5)}")
