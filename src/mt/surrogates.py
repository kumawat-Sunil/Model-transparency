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
