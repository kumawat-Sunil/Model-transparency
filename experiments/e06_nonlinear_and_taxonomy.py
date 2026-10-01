"""E06a: classification tasks where a black box beats logistic regression: does the symbolic surrogate keep the gain?
E06b: expressibility/cost of harder preprocessing steps (PCA, Polynomial, Quantile, KBins) when compiled to closed form."""
import sys, pathlib, warnings
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PolynomialFeatures, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from mt.surrogates import sparse_select, gp_fit, gp_to_sympy
from mt.transforms import n_nodes
OUT = ROOT / "results" / "e06"; OUT.mkdir(parents=True, exist_ok=True)
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))); sig = lambda x: 1 / (1 + np.exp(-x))
rng = np.random.default_rng(0)
def pad(X, n=3): return np.hstack([X, rng.normal(size=(len(X), n))])  # irrelevant noise features
tasks = {}
X, y = make_moons(2000, noise=0.25, random_state=0); tasks["moons+3noise"] = (pad(X), y)
X, y = make_circles(2000, noise=0.12, factor=0.5, random_state=0); tasks["circles+3noise"] = (pad(X), y)
X = rng.normal(size=(2000, 6)); y = (X[:, 0] * X[:, 1] + 0.5 * X[:, 2] ** 2 - 0.3 + 0.3 * rng.normal(size=2000) > 0).astype(int); tasks["xor-quad(6d)"] = (X, y)
X = rng.normal(size=(2000, 8)); s = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] - np.abs(X[:, 3]) + 0.3 * rng.normal(size=2000); tasks["sin-abs-prod(8d)"] = (X, (s > 0).astype(int))
rows = []
for tn, (X, y) in tasks.items():
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y); Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
    sc = StandardScaler().fit(Xtr); Ztr, Zva, Zte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte); k = Ztr.shape[1]
    def rec(m, p, **kw): rows.append(dict(task=tn, method=m, auc=roc_auc_score(yte, p), acc=accuracy_score(yte, p > .5), **kw))
    rec("logreg", LogisticRegression().fit(Ztr, ytr).predict_proba(Zte)[:, 1], complexity=k + 1)
    for nm, t in {"xgb": xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0),
                  "mlp": MLPClassifier((64, 64), activation="tanh", max_iter=3000, random_state=0, alpha=1e-2)}.items():
        t.fit(Ztr, ytr); pt = t.predict_proba(Zte)[:, 1]; rec(f"teacher_{nm}", pt)
        lq, lv = logit(t.predict_proba(Ztr)[:, 1]), logit(t.predict_proba(Zva)[:, 1])
        g, nt, path = sparse_select(Ztr, lq, Zva, lv, return_path=True)
        for npar, _, ex in [(nt, 0, g)] + [p for p in path if p[0] in (3, 6)]:
            gf = sp.lambdify(sp.symbols(f"z0:{k}"), ex, "numpy"); ps = sig(np.asarray(gf(*Zte.T), float) * np.ones(len(Zte)))
            rec(f"sparse<-{nm}", ps, terms=npar, complexity=n_nodes(ex), fid_agree=float(np.mean((ps > .5) == (pt > .5))))
        sub = rng.choice(len(Ztr), min(1000, len(Ztr)), replace=False)
        sr = gp_fit(Ztr[sub], lq[sub], 1e-3, pop=800, gens=20, funcs=("add", "sub", "mul", "div", "sin", "cos", "sqrt")); g2, s2 = gp_to_sympy(sr, k)
        ps = sig(sr.predict(Zte)); rec(f"gp<-{nm}", ps, complexity=n_nodes(g2), fid_agree=float(np.mean((ps > .5) == (pt > .5))))
    print(tn, flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "classification.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200); print(df.round(3).to_string())

# ---------------- E06b: preprocessing taxonomy ----------------
Xp = np.random.default_rng(1).lognormal(size=(1000, 6)); Xq = Xp[:200]; syms = sp.symbols("x0:6"); tx = []
def exact_check(name, fitted, exprs, X):
    fn = [sp.lambdify(syms, e, "numpy") for e in exprs]; Zs = np.column_stack([np.asarray(f(*X.T), float) * np.ones(len(X)) for f in fn])
    tx.append(dict(step=name, n_out=len(exprs), nodes=sum(n_nodes(e) for e in exprs), max_abs_err=float(np.max(np.abs(Zs - fitted.transform(X)))), closed_form=name in ("PCA", "Polynomial(2)")))
sc = StandardScaler().fit(Xp); pca = PCA(4).fit(sc.transform(Xp))
zs = [(syms[i] - sp.Float(sc.mean_[i], 17)) / sp.Float(sc.scale_[i], 17) for i in range(6)]
class _Pipe:  # sklearn-like facade so exact_check works for composite
    def __init__(s, f): s.f = f
    def transform(s, X): return s.f(X)
exact_check("PCA", _Pipe(lambda X: pca.transform(sc.transform(X))), [sum(sp.Float(pca.components_[j, i], 17) * zs[i] for i in range(6)) - sp.Float(float(pca.mean_ @ pca.components_[j]), 17) for j in range(4)], Xq)
pf = PolynomialFeatures(2, include_bias=False).fit(Xp[:, :3]); import itertools
exprs = [syms[0], syms[1], syms[2]] + [syms[i] * syms[j] for i, j in itertools.combinations_with_replacement(range(3), 2)]
exact_check("Polynomial(2)", _Pipe(lambda X: pf.transform(X[:, :3])), exprs, Xq)
for nq in (10, 50, 200):
    qt = QuantileTransformer(n_quantiles=nq, output_distribution="uniform").fit(Xp)
    ex = []
    for i in range(6):  # piecewise-linear interpolation through (quantile, ref) knots == sklearn's definition
        q, r = qt.quantiles_[:, i], qt.references_
        ex.append(sp.Piecewise(*[((sp.Float(r[j], 17) + (syms[i] - sp.Float(q[j], 17)) * sp.Float((r[j + 1] - r[j]) / max(q[j + 1] - q[j], 1e-300), 17)), syms[i] < sp.Float(q[j + 1], 17)) for j in range(nq - 1)], (1, True)))
    exact_check(f"Quantile(n={nq})", _Pipe(lambda X, qt=qt: qt.transform(X)), ex, Xq)
tx = pd.DataFrame(tx); tx.to_csv(OUT / "taxonomy.csv", index=False); print(tx.to_string())
