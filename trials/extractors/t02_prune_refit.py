"""t02 = t01 + (1) 3 Operon seeds, (2) greedy pruning: drop Add-terms or freeze sub-trees (sin/cos/exp/log/pow) to a constant
when held-out fidelity drops < tol, (3) least-squares refit of all constants after every edit, (4) round constants to 4 sig figs."""
import numpy as np, sympy as sp
from scipy.optimize import least_squares
from pyoperon.sklearn import SymbolicRegressor
from extractors.t01_operon_multirep import Model2Formula as Base, OPS

def refit(e, S, F, y):
    consts = [a for a in e.atoms(sp.Float)]
    if not consts: return e
    P = sp.symbols(f"c0:{len(consts)}"); ep = e.xreplace(dict(zip(consts, P)))
    f = sp.lambdify(list(S) + list(P), ep, "numpy")
    def res(c):
        with np.errstate(all="ignore"): r = np.asarray(f(*F.T, *c), float) * np.ones(len(F)) - y
        return np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6)
    try: sol = least_squares(res, np.array([float(c) for c in consts]), max_nfev=200)
    except Exception: return e
    return ep.xreplace({p: sp.Float(v, 6) for p, v in zip(P, sol.x)})

def fid(e, S, F, y):
    f = sp.lambdify(S, e, "numpy")
    with np.errstate(all="ignore"): p = np.asarray(f(*F.T), float) * np.ones(len(F))
    return 1 - np.mean((p - y) ** 2) / np.var(y) if np.all(np.isfinite(p)) else -np.inf

def candidates(e):
    out = []
    if isinstance(e, sp.Add):
        for t in e.args: out.append(e - t)
    for sub in sp.preorder_traversal(e):
        if sub is e or sub.is_Atom: continue
        if isinstance(sub, (sp.sin, sp.cos, sp.exp, sp.log, sp.Pow)) or (isinstance(sub, sp.Mul) and sub.has(sp.Function)):
            out.append(e.xreplace({sub: sp.Float(1.0)}))
    return out

class Model2Formula(Base):
    def __init__(self, *a, tol=0.003, seeds=(0, 1, 2), **k): super().__init__(*a, **k); self.tol, self.seeds = tol, seeds

    def fit(self, X_raw):
        best_overall = None
        for sd in self.seeds:
            self.seed = sd; super().fit(X_raw)
            if best_overall is None or max(c["score"] for c in self.candidates) > best_overall[0]:
                best_overall = (max(c["score"] for c in self.candidates), self.formula, self.choice)
        _, e, self.choice = best_overall
        # rebuild query data (same recipe as base) for polishing
        rng = np.random.default_rng(99); idx = rng.integers(0, len(X_raw), (self.n_query, X_raw.shape[1]))
        Xq = np.vstack([X_raw, X_raw[idx, np.arange(X_raw.shape[1])]]); F, y = self.T(Xq), self._model_out(Xq)
        ok = np.all(np.isfinite(F), 1) & np.isfinite(y); F, y = F[ok], y[ok]
        S = [sp.Symbol(n) for n in self.pre]; e = refit(e, S, F, y); cur = fid(e, S, F, y); self.raw_before_prune = e
        improved = True
        while improved:
            improved = False
            for c in sorted(candidates(e), key=lambda q: sp.count_ops(q)):
                c = refit(c, S, F, y); fc = fid(c, S, F, y)
                if fc >= cur - self.tol and sp.count_ops(c) < sp.count_ops(e):
                    e, cur, improved = c, fc, True; break
        e = e.xreplace({c: sp.Float(float(sp.N(c, 4)), 4) for c in e.atoms(sp.Float)})
        self.formula = e; self.choice = (*self.choice, f"after-prune fid={cur:.4f}")
        self.full_formula = e.xreplace({sp.Symbol(n): x for n, x in zip(self.pre, self.T_sym())})
        if self.logit: self.full_formula = 1 / (1 + sp.exp(-self.full_formula))
        self._f = sp.lambdify([sp.Symbol(n) for n in self.raw], self.full_formula, "numpy"); return self
