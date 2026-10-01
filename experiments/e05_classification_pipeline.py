"""E05: classification with a realistic preprocessing pipeline (imputation, log1p, scaling, one-hot) -> teacher (XGB/MLP) -> symbolic
logit surrogate -> sigmoid(g(T(raw))) on raw columns. Tests (a) exact compilation of T, (b) fidelity/accuracy/complexity, (c) the breast-cancer
dataset (30 numeric features) as a higher-dimensional case."""
import sys, pathlib, json, warnings, time
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.datasets import load_breast_cancer
import xgboost as xgb
from mt.compile_sklearn import compile_column_transformer, NUMPY_MODS
from mt.surrogates import sparse_select
from mt.fuse import fuse
from mt.transforms import n_nodes

OUT = ROOT / "results" / "e05"; OUT.mkdir(parents=True, exist_ok=True)
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
sig = lambda x: 1 / (1 + np.exp(-x))

def load_titanic():
    d = pd.read_csv(ROOT / "data/titanic.csv")
    d["sex"] = (d.sex == "female").astype(float)                       # string->code dictionary: lookup, outside the closed form
    d["embarked"] = d.embarked.map({"S": 0.0, "C": 1.0, "Q": 2.0})       # NaN stays NaN
    X = d[["age", "fare", "sibsp", "parch", "pclass", "sex", "embarked"]].astype(float); return X, d.survived.values, \
        ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), ["age", "sibsp", "parch"]),
                           ("fare", Pipeline([("imp", SimpleImputer(strategy="median")), ("log", FunctionTransformer(np.log1p)), ("sc", StandardScaler())]), ["fare"]),
                           ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["pclass", "sex", "embarked"])])

def load_bc():
    b = load_breast_cancer(as_frame=True); X = b.data.copy(); X.columns = [c.replace(" ", "_") for c in X.columns]
    return X, b.target.values, ColumnTransformer([("num", Pipeline([("log", FunctionTransformer(np.log1p)), ("sc", StandardScaler())]), list(X.columns))])

rows = []; formulas = {}
for dname, loader in [("titanic", load_titanic), ("breast_cancer", load_bc)]:
    X, y, ct = loader()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y); Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
    ct.fit(Xtr); Ztr, Zva, Zte = ct.transform(Xtr), ct.transform(Xva), ct.transform(Xte); k = Ztr.shape[1]
    names, zex = compile_column_transformer(ct, list(X.columns))
    # (a) exactness of compiled T
    fns = [sp.lambdify([sp.Symbol(c) for c in X.columns], e, modules=NUMPY_MODS) for e in zex]
    Zsym = np.column_stack([np.asarray(f(*[Xte[c].values for c in X.columns]), float) * np.ones(len(Xte)) for f in fns])
    t_err = float(np.max(np.abs(Zsym - Zte)))
    rows.append(dict(dataset=dname, method="T_compile_check", complexity=sum(n_nodes(e) for e in zex), max_abs_err=t_err, n_features=k))
    teachers = {"xgb": xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0),
                "mlp": MLPClassifier((64, 64), activation="tanh", max_iter=2000, random_state=0, alpha=1e-2)}
    lr = LogisticRegression(max_iter=5000, C=1.0).fit(Ztr, ytr); pl = lr.predict_proba(Zte)[:, 1]
    rows.append(dict(dataset=dname, method="logreg_baseline", complexity=k + 1, auc=roc_auc_score(yte, pl), acc=accuracy_score(yte, pl > .5)))
    for tn, t in teachers.items():
        t.fit(Ztr, ytr); pt = t.predict_proba(Zte)[:, 1]
        rows.append(dict(dataset=dname, method=f"teacher_{tn}", auc=roc_auc_score(yte, pt), acc=accuracy_score(yte, pt > .5)))
        for qn, (Zq, lq) in {"distill": (Ztr, logit(t.predict_proba(Ztr)[:, 1])),
                             "labels(logit-ls)": (Ztr, np.where(ytr == 1, 2.5, -2.5).astype(float))}.items():
            if qn.startswith("labels") and tn == "mlp": continue
            g, nt, path = sparse_select(Zq, lq, Zva, logit(t.predict_proba(Zva)[:, 1]) if qn == "distill" else np.where(yva == 1, 2.5, -2.5).astype(float), rule="min", return_path=True)
            for npar, _, ex in ([(nt, 0, g)] + [p for p in path if p[0] in (2, 4, 8)]):
                gf = sp.lambdify([sp.Symbol(f"z{i}") for i in range(k)], ex, "numpy")
                ps = sig(np.asarray(gf(*Zte.T), float) * np.ones(len(Zte)))
                F = fuse(ex, zex)
                Ff = sp.lambdify([sp.Symbol(c) for c in X.columns], 1 / (1 + sp.exp(-F)), modules=NUMPY_MODS)
                pF = np.asarray(Ff(*[Xte[c].values for c in X.columns]), float) * np.ones(len(Xte))
                rows.append(dict(dataset=dname, method=f"sym[{qn}]<-{tn}", terms=npar, complexity=n_nodes(ex), fused_nodes=n_nodes(F),
                                 auc=roc_auc_score(yte, ps), acc=accuracy_score(yte, ps > .5), fid_agree=float(np.mean((ps > .5) == (pt > .5))),
                                 fused_max_err=float(np.max(np.abs(pF - ps)))))
    print(dname, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "metrics.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200); print(df.round(4).to_string())
