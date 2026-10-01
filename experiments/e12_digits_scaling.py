"""E12: scaling to 64 inputs. digits 3-vs-8, MLP/XGB teacher, sparse symbolic logit (linear+deg2 library = 2144 terms)."""
import sys, pathlib, warnings; ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp, time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from mt.surrogates import sparse_select
from mt.transforms import n_nodes
d = load_digits(); m = np.isin(d.target, [3, 8]); X, y = d.data[m] / 16.0, (d.target[m] == 8).astype(int)
keep = X.std(0) > 0; X = X[:, keep]; k = X.shape[1]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y); Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Z = lambda A: (A - mu) / sd
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))); sig = lambda x: 1 / (1 + np.exp(-x)); rows = []
rows.append(dict(method="logreg", auc=roc_auc_score(yte, LogisticRegression(max_iter=3000).fit(Z(Xtr), ytr).predict_proba(Z(Xte))[:, 1]), nodes=k + 1))
for nm, t in {"mlp": MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, random_state=0, alpha=1e-2), "xgb": xgb.XGBClassifier(n_estimators=200, max_depth=3, random_state=0)}.items():
    t.fit(Z(Xtr), ytr); pt = t.predict_proba(Z(Xte))[:, 1]; rows.append(dict(method=f"teacher_{nm}", auc=roc_auc_score(yte, pt), acc=accuracy_score(yte, pt > .5)))
    for deg2 in (False, True):
        t0 = time.time(); g, nt, path = sparse_select(Z(Xtr), logit(t.predict_proba(Z(Xtr))[:, 1]), Z(Xva), logit(t.predict_proba(Z(Xva))[:, 1]), deg2=deg2, return_path=True)
        for npar, _, ex in [(nt, 0, g)] + [p for p in path if p[0] in (5, 10, 20)]:
            f = sp.lambdify(sp.symbols(f"z0:{k}"), ex, "numpy"); ps = sig(np.asarray(f(*Z(Xte).T), float) * np.ones(len(Xte)))
            rows.append(dict(method=f"sparse{'-deg2' if deg2 else '-lin'}<-{nm}", terms=npar, nodes=n_nodes(ex), auc=roc_auc_score(yte, ps), acc=accuracy_score(yte, ps > .5), agree=float(np.mean((ps > .5) == (pt > .5))), secs=time.time() - t0))
df = pd.DataFrame(rows); (ROOT / "results/e12").mkdir(exist_ok=True); df.to_csv(ROOT / "results/e12/metrics.csv", index=False); pd.set_option("display.width", 200); print(df.round(3).to_string())
