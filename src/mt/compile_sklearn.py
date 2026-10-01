"""Compile a fitted sklearn ColumnTransformer (imputer / log1p / scaler / one-hot / PCA-free subset) into sympy expressions of raw columns.
Expressibility taxonomy (H5):
  exact algebraic : StandardScaler, MinMax, log1p, polynomial, PCA (linear map)
  exact piecewise : SimpleImputer (Piecewise on isnan), OneHot on integer codes (indicator)
  NOT closed form : string->code dictionaries, quantile/rank transforms (tables), target encoding (tables)  -> kept as lookup tables outside F
"""
import numpy as np, sympy as sp
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

isnan = sp.Function("isnan")
NUMPY_MODS = [{"isnan": np.isnan}, "numpy"]


def _apply(step, exprs):
    if isinstance(step, SimpleImputer):
        return [sp.Piecewise((sp.Float(float(step.statistics_[i]), 17), sp.Eq(isnan(e), 1)), (e, True)) for i, e in enumerate(exprs)]
    if isinstance(step, StandardScaler):
        return [(e - sp.Float(float(step.mean_[i]), 17)) / sp.Float(float(step.scale_[i]), 17) for i, e in enumerate(exprs)]
    if isinstance(step, FunctionTransformer):
        if step.func is np.log1p: return [sp.log(1 + e) for e in exprs]
        raise NotImplementedError(step.func)
    if isinstance(step, OneHotEncoder):
        out = []
        for e, cats in zip(exprs, step.categories_):
            out += [sp.Piecewise((1, sp.Eq(e, sp.Float(float(c)))), (0, True)) for c in cats]
        return out
    raise NotImplementedError(type(step))


def compile_column_transformer(ct, raw_names):
    names, exprs = [], []
    for nm, trans, cols in ct.transformers_:
        if nm == "remainder": continue
        cur = [sp.Symbol(c) for c in cols]
        steps = [s for _, s in trans.steps] if isinstance(trans, Pipeline) else [trans]
        for s in steps: cur = _apply(s, cur)
        exprs += cur; names += [f"{nm}_{i}" for i in range(len(cur))]
    return names, exprs
