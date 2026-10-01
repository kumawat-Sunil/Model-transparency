"""Surrogate searchers. All operate in z-space (standardized engineered features) and return a sympy expr in symbols z0..zk."""
import itertools, numpy as np, sympy as sp
from sklearn.linear_model import Lasso, LinearRegression
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from .transforms import n_nodes


def zsyms(k): return [sp.Symbol(f"z{i}") for i in range(k)]


# ---- A: sparse regression over a term library (SINDy/STLSQ-style, quantified by sparsity) ----
def library(Z, deg2=True, trig=True):
    k = Z.shape[1]; S = zsyms(k)
    terms = [(s, Z[:, i]) for i, s in enumerate(S)]
    if deg2:
        for i, j in itertools.combinations_with_replacement(range(k), 2):
            terms.append((S[i] * S[j], Z[:, i] * Z[:, j]))
    return terms


def sparse_fit(Z, y, alpha, deg2=True):
    terms = library(Z, deg2)
    A = np.column_stack([t[1] for t in terms])
    sc = A.std(0) + 1e-12
    m = Lasso(alpha=alpha, max_iter=50000).fit(A / sc, y)
    keep = np.abs(m.coef_) > 1e-10
    if keep.sum() == 0:
        return sp.Float(float(y.mean())), 1
    ols = LinearRegression().fit(A[:, keep], y)  # debias
    expr = sp.Float(float(ols.intercept_), 8)
    for c, (s, _), kp in zip(ols.coef_, [t for t, k in zip(terms, keep) if k], [1] * keep.sum()):
        expr += sp.Float(float(c), 8) * s
    return expr, int(keep.sum())


def sparse_path(Z, y, alphas, deg2=True):
    return [(a,) + sparse_fit(Z, y, a, deg2) for a in alphas]


# ---- B: genetic-programming SR (gplearn) ----
def gp_fit(Z, y, parsimony, seed=0, pop=1500, gens=25, funcs=("add", "sub", "mul", "div", "log", "sqrt", "sin", "cos")):
    sr = SymbolicRegressor(population_size=pop, generations=gens, function_set=funcs, parsimony_coefficient=parsimony,
                           p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, const_range=(-5, 5), init_depth=(2, 5), random_state=seed, n_jobs=4, verbose=0)
    sr.fit(Z, y)
    return sr


def gp_to_sympy(sr, k):
    S = zsyms(k)
    s = str(sr._program)
    loc = {f"X{i}": S[i] for i in range(k)}
    loc.update({"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
                "div": lambda a, b: a / b if b != 0 else sp.Integer(1),
                "sqrt": lambda a: sp.sqrt(sp.Abs(a)), "log": lambda a: sp.log(sp.Abs(a)),
                "sin": sp.sin, "cos": sp.cos, "neg": lambda a: -a, "abs": sp.Abs})
    return sp.sympify(s, locals=loc), s


# ---- C: exact tree ensemble -> expression size accounting (no sympy blowup: count nodes analytically) ----
def tree_node_count(model, kind):
    if kind == "xgb":
        df = model.get_booster().trees_to_dataframe()
        return int(len(df))
    if kind == "lgbm":
        df = model.booster_.trees_to_dataframe()
        return int(len(df))
    raise ValueError


# ---- A2: repaired sparse engine: lasso path -> ridge refit on support -> pick support size by validation ----
from sklearn.linear_model import Ridge, lasso_path

def _prune_collinear(A, tol=0.995):
    keep = []
    C = np.corrcoef(A.T)
    for j in range(A.shape[1]):
        if all(abs(C[j, i]) < tol for i in keep): keep.append(j)
    return keep


def sparse_select(Z, y, Zv, yv, deg2=True, n_alphas=25, ridge=1e-3, rule="min", return_path=False):
    """Return (expr, n_terms) where support chosen on validation RMSE against yv (the labels, for fairness across methods)."""
    terms = library(Z, deg2)
    A = np.column_stack([t[1] for t in terms]); Av = np.column_stack([library(Zv, deg2)[i][1] for i in range(len(terms))])
    kp = _prune_collinear(A); A, Av = A[:, kp], Av[:, kp]; terms = [terms[i] for i in kp]
    mu, sc = A.mean(0), A.std(0) + 1e-12
    As, Avs = (A - mu) / sc, (Av - mu) / sc
    alphas, coefs, _ = lasso_path(As, y - y.mean(), n_alphas=n_alphas, eps=1e-4)
    best = None; seen = set(); path = []
    for j in range(coefs.shape[1]):
        sup = tuple(np.flatnonzero(np.abs(coefs[:, j]) > 1e-10))
        if not sup or sup in seen: continue
        seen.add(sup)
        m = Ridge(alpha=ridge).fit(As[:, sup], y)
        res = m.predict(Avs[:, sup]) - yv
        e = float(np.sqrt(np.mean(res ** 2))); se = float(np.std(res ** 2) / np.sqrt(len(res)) / (2 * e + 1e-12))
        path.append((len(sup), e, se, sup, m))
        if best is None or e < best[0] - 1e-9: best = (e, sup, m)
    e, sup, m = best
    if rule == "1se":  # smallest support within one standard error of the best validation RMSE
        thr = e + max(p[2] for p in path if p[3] == sup)
        ok = [p for p in path if p[1] <= thr]; _, e, _, sup, m = min(ok, key=lambda p: p[0])
    expr = sp.Float(float(m.intercept_ - np.sum(m.coef_ * mu[list(sup)] / sc[list(sup)])), 8)
    for c, i in zip(m.coef_, sup):
        expr += sp.Float(float(c / sc[i]), 8) * terms[i][0]
    if return_path:
        def build(sup_, m_):
            ex = sp.Float(float(m_.intercept_ - np.sum(m_.coef_ * mu[list(sup_)] / sc[list(sup_)])), 8)
            for c, i in zip(m_.coef_, sup_): ex += sp.Float(float(c / sc[i]), 8) * terms[i][0]
            return ex
        return expr, len(sup), [(n, e_, build(s_, m_)) for n, e_, _, s_, m_ in path]
    return expr, len(sup)


def sparse_cv_select(Z, y_target, y_true, fold_origins, horizon=12, deg2=True, sizes=(1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30), ridge=1e-3, return_cv=False):
    """Rolling-origin CV choice of support size (rows of Z assumed time-ordered). Supports from lasso path on full train;
    per fold ridge-refit on rows < origin, score on next `horizon` rows against TRUE labels; pick smallest size within 1 SE of best CV mean."""
    terms = library(Z, deg2); A = np.column_stack([t[1] for t in terms])
    kp = _prune_collinear(A); A = A[:, kp]; terms = [terms[i] for i in kp]
    mu, sc = A.mean(0), A.std(0) + 1e-12; As = (A - mu) / sc
    alphas, coefs, _ = lasso_path(As, y_target - y_target.mean(), n_alphas=60, eps=1e-4)
    sup_by_size = {}
    for j in range(coefs.shape[1]):
        sup = tuple(np.flatnonzero(np.abs(coefs[:, j]) > 1e-10))
        for s in sizes:
            if len(sup) >= s and (s not in sup_by_size or len(sup) < len(sup_by_size[s])): sup_by_size[s] = sup
    cv = {}
    for s, sup in sup_by_size.items():
        errs = []
        for o in fold_origins:
            m = Ridge(alpha=ridge).fit(As[:o][:, sup], y_target[:o]); p = m.predict(As[o:o + horizon][:, sup])
            errs.append(np.sqrt(np.mean((p - y_true[o:o + horizon]) ** 2)))
        cv[s] = (float(np.mean(errs)), float(np.std(errs) / np.sqrt(len(errs))))
    best = min(cv, key=lambda s: cv[s][0]); thr = cv[best][0] + cv[best][1]
    chosen = min([s for s in cv if cv[s][0] <= thr])
    sup = sup_by_size[chosen]; m = Ridge(alpha=ridge).fit(As[:, sup], y_target)
    expr = sp.Float(float(m.intercept_ - np.sum(m.coef_ * mu[list(sup)] / sc[list(sup)])), 8)
    for c, i in zip(m.coef_, sup): expr += sp.Float(float(c / sc[i]), 8) * terms[i][0]
    return (expr, len(sup), cv) if return_cv else (expr, len(sup))
