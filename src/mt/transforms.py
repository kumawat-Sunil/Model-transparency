"""Preprocessing pipeline whose every step is also a sympy expression.

SymbolicFeatures: raw symbols -> engineered feature expressions (lags mean, log, cyclic month ...)
Standardizer: fitted mu/sigma, exposed symbolically.
Together they give T: raw -> z as a list of sympy expressions, and a numeric twin for the teacher.
"""
import numpy as np, sympy as sp


class SymbolicPipeline:
    def __init__(self, raw_names, feature_exprs, standardize=True):
        """feature_exprs: dict name -> function(dict_of_symbols_or_arrays, lib) -> value.
        The SAME callable is run with numpy (numeric) and sympy (symbolic) so the two cannot diverge."""
        self.raw_names = list(raw_names)
        self.feature_exprs = feature_exprs
        self.standardize = standardize
        self.names = list(feature_exprs)

    # numeric path
    def _raw_features(self, X):
        env = {n: X[:, i] for i, n in enumerate(self.raw_names)}
        return np.column_stack([f(env, np) for f in self.feature_exprs.values()])

    def fit(self, X):
        F = self._raw_features(X)
        self.mu_ = F.mean(0); self.sd_ = F.std(0) + 1e-12
        return self

    def transform(self, X):
        F = self._raw_features(X)
        return (F - self.mu_) / self.sd_ if self.standardize else F

    # symbolic path
    def symbols(self):
        return {n: sp.Symbol(n) for n in self.raw_names}

    def z_exprs(self):
        env = self.symbols()
        out = []
        for i, f in enumerate(self.feature_exprs.values()):
            e = f(env, _SymLib)
            if self.standardize:
                e = (e - sp.Float(self.mu_[i], 17)) / sp.Float(self.sd_[i], 17)
            out.append(e)
        return out


class _SymLib:  # numpy-like facade over sympy
    log = staticmethod(sp.log); exp = staticmethod(sp.exp); sin = staticmethod(sp.sin)
    cos = staticmethod(sp.cos); sqrt = staticmethod(sp.sqrt); pi = sp.pi
    log1p = staticmethod(lambda x: sp.log(1 + x))
    maximum = staticmethod(sp.Max)


def n_nodes(expr):
    return sum(1 for _ in sp.preorder_traversal(expr))
