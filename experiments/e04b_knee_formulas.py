"""E04b: print the human-readable fused formulas at the Pareto knee (distilled from XGB)."""
import sys, pathlib, warnings
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp
sys.argv = ["x"]; exec(open(ROOT / "experiments/e03_real_forecasting.py").read().split("rows = []")[0])  # reuse loader/recursive
from mt.data import make_real_pipeline, window_xy, RAW_REAL
from mt.teachers import make_teacher
from mt.surrogates import sparse_select
from mt.fuse import fuse
out = ["# Knee formulas (XGB teacher -> sparse symbolic -> fused with T; raw symbols l1,l2,l3,l6,l12,month)\n"]
for name, nterms in [("airline", 3), ("co2", 5), ("sunspots", 3), ("carsales", 4)]:
    y, months = load()[name]; Hn = 24 if name != "sunspots" else 60; n = len(y); idx, X, T = window_xy(y, months)
    te = idx >= n - Hn; va = (idx >= n - 2 * Hn) & ~te; tr = idx < n - 2 * Hn
    pipe = make_real_pipeline().fit(X[tr]); Z = {k: pipe.transform(X[m]) for k, m in dict(tr=tr, va=va).items()}
    t = make_teacher("xgb", 0).fit(Z["tr"], T[tr])
    _, _, path = sparse_select(Z["tr"], t.predict(Z["tr"]), Z["va"], T[va], return_path=True)
    g = [e for nt, _, e in path if nt == nterms][0]
    names = pipe.names; g_named = g.subs({sp.Symbol(f"z{i}"): sp.Symbol(nm) for i, nm in enumerate(names)})
    F = sp.simplify(sp.N(fuse(g, pipe.z_exprs()), 5))
    out += [f"## {name} ({nterms} terms)\n", f"In feature space: `{sp.N(g_named, 4)}`\n", f"Fused on raw: `{F}`\n"]
open(ROOT / "results/e04/knee_formulas.md", "w").write("\n".join(out)); print("\n".join(out))
