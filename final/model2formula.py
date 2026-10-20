"""FINAL: trained ML model + preprocessing  ->  one closed-form formula on RAW inputs (+ domain guard + standalone export).

    from model2formula import extract
    res = extract(model, preprocess, raw_names, X_train_raw, problem_type, model_input="features"|"raw", classification=False)
    res.full_formula        # sympy formula in raw inputs
    res.predict(X_raw)      # formula-only prediction
    res.in_domain(X_raw)    # True where inputs lie inside the training range (trust region)
    res.export("file.py")   # standalone numpy file

problem_type -> method (chosen after verification on 3-5 datasets per type, see README.md):
    physical_law        : t06  Operon multi-representation search, fidelity-first, prune + refit + snap, safe-op rule
    forecasting         : t06  (+ periodic-feature rule: no trig on trend features)
    tabular_regression  : best SANE formula of {t08 additive, t06 search, t09 sparse polynomial} on fresh validation queries (finite outputs required)
    classification      : same 3-candidate sane selection (fits the logit, returns a probability formula)
preprocess: dict feature_name -> lambda env, L: expression using L.log / L.sin / L.cos / L.exp / L.sqrt / L.pi (same code is
used numerically and symbolically, so the preprocessing inside the formula is exact)."""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from engine.t06_fidelity_first import Model2Formula as Search
from engine.t08_additive import Model2Formula as Additive
from engine.t09_sparse_candidate import Model2Formula as SparsePoly
from engine.t09_sanity import sane
import sympy as sp

def _val_fidelity(ex, X_raw):
    F_rows = np.vstack([X_raw, X_raw[np.random.default_rng(7).integers(0, len(X_raw), (3000, X_raw.shape[1])), np.arange(X_raw.shape[1])]])
    with np.errstate(all="ignore"):
        p, m = ex.predict(F_rows), ex.model_predict(F_rows)
    if not np.all(np.isfinite(p)): return -np.inf
    return 1 - np.mean((p - m) ** 2) / np.var(m)

def extract(model, preprocess, raw_names, X_train_raw, problem_type, model_input="features", classification=False):
    kw = dict(model_input=model_input, logit=classification)
    if problem_type in ("physical_law", "forecasting"):
        res = Search(model, preprocess, raw_names, **kw).fit(X_train_raw); res.method = "t06"; return res
    cands = []
    for name, cls in (("t08_additive", Additive), ("t06_search", Search), ("t09_sparse_poly", SparsePoly)):
        try:
            ex = cls(model, preprocess, raw_names, **kw).fit(X_train_raw); ex.method = name
            S = [sp.Symbol(n) for n in preprocess]; ok, why = sane(ex.formula, S, ex.T(X_train_raw))
            cands.append((_val_fidelity(ex, X_train_raw), ok, why, ex))
        except Exception as err:
            print("candidate failed:", name, err)
    pool = [c for c in cands if c[1]] or cands          # v2: only numerically sane formulas compete (fallback: all)
    best = max(pool, key=lambda c: c[0])[3]; best.validation = {c[3].method: (round(float(c[0]), 4), c[2]) for c in cands}
    return best
