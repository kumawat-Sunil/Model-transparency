import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import numpy as np, sympy as sp
from mt.data import make_series, make_pipeline, RAW
from mt.fuse import fuse, to_callable, eval_on_raw

def test_symbolic_pipeline_matches_numeric_exactly():
    d = make_series(5, 40, seed=1); X = d["X"]
    p = make_pipeline().fit(X)
    Z_num = p.transform(X)
    fns = [to_callable(e, RAW) for e in p.z_exprs()]
    Z_sym = np.column_stack([eval_on_raw(f, X) for f in fns])
    assert np.max(np.abs(Z_num - Z_sym)) < 1e-9

def test_fuse_equals_composition():
    d = make_series(5, 40, seed=2); X = d["X"]; p = make_pipeline().fit(X)
    S = [sp.Symbol(f"z{i}") for i in range(len(p.names))]
    g = 2 * S[0] + sp.sin(S[5]) * S[1] - S[3] ** 2
    F = fuse(g, p.z_exprs()); fF = to_callable(F, RAW)
    Z = p.transform(X); ref = 2 * Z[:, 0] + np.sin(Z[:, 5]) * Z[:, 1] - Z[:, 3] ** 2
    assert np.max(np.abs(eval_on_raw(fF, X) - ref)) < 1e-9
