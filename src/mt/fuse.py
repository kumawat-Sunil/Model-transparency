"""Fuse symbolic model g(z) with symbolic pipeline T(raw) into F(raw); verify numerically."""
import numpy as np, sympy as sp
from .transforms import n_nodes


def fuse(g_expr, z_exprs, simplify=False):
    sub = {sp.Symbol(f"z{i}"): e for i, e in enumerate(z_exprs)}
    F = g_expr.xreplace(sub)
    return sp.simplify(F) if simplify else F


def to_callable(F, raw_names):
    return sp.lambdify([sp.Symbol(n) for n in raw_names], F, modules="numpy")


def eval_on_raw(fn, X):
    return np.asarray(fn(*[X[:, i] for i in range(X.shape[1])]), dtype=float) * np.ones(len(X))
