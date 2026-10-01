"""t03 = t02 + (1) two operator sets: full and SAFE (add,sub,mul,div,log,sqrt,square: monotone-ish, no oscillation),
(2) selection score = min(fidelity on random held-out, fidelity on EDGE rows (any feature in its outer 10% quantiles)) - 0.0005*nodes,
(3) snapping: exponents/constants -> nearest multiple of 0.5 (or integer coefficient) accepted if fidelity drop < tol, then refit others."""
import numpy as np, sympy as sp
from pyoperon.sklearn import SymbolicRegressor
from engine.t02_prune_refit import refit, fid, candidates
import engine.t01_operon_multirep as t01

SAFE = "add,sub,mul,div,constant,variable,log,sqrt,square"

class Model2Formula(t01.Model2Formula):
    def __init__(self, *a, tol=0.003, seeds=(0, 1), **k): super().__init__(*a, **k); self.tol, self.seeds = tol, seeds

    def _queries(self, X_raw, seed):
        rng = np.random.default_rng(seed); idx = rng.integers(0, len(X_raw), (self.n_query, X_raw.shape[1]))
        Xq = np.vstack([X_raw, X_raw[idx, np.arange(X_raw.shape[1])]]); F, y = self.T(Xq), self._model_out(Xq)
        ok = np.all(np.isfinite(F), 1) & np.isfinite(y); return F[ok], y[ok]

    def fit(self, X_raw):
        F, y = self._queries(X_raw, 0); S = [sp.Symbol(n) for n in self.pre]
        lo, hi = np.quantile(F, 0.1, 0), np.quantile(F, 0.9, 0); edge = np.any((F < lo) | (F > hi), 1)
        rng = np.random.default_rng(1); perm = rng.permutation(len(F)); tr, ho = perm[: int(.8 * len(F))], perm[int(.8 * len(F)):]
        ho_edge = ho[edge[ho]]
        loc = {f"X{i+1}": S[i] for i in range(len(S))}; loc["square"] = lambda q: q ** 2
        reps = {"raw": (lambda Z: Z, None)}
        if np.all(F > 0): reps["log-inputs"] = (np.log, "li")
        if np.all(F > 0) and np.all(y > 0): reps["log-log"] = (np.log, "ll")
        score = lambda e: min(fid(e, S, F[ho], y[ho]), fid(e, S, F[ho_edge], y[ho_edge])) - getattr(self, 'penalty', 0.0005) * sum(1 for _ in sp.preorder_traversal(e))
        self.candidates = []
        for opsname, ops in (("full", t01.OPS), ("safe", SAFE)):
            for rn, (fx, kind) in reps.items():
                ty = np.log(y) if kind == "ll" else y
                for L in getattr(self, 'lengths', (10, 20)):
                    for sd in self.seeds:
                        sr = SymbolicRegressor(allowed_symbols=ops, generations=getattr(self, 'gens', 200), population_size=getattr(self, 'pop', 1000), max_length=L, n_threads=4, random_state=sd).fit(fx(F[tr]), ty[tr])
                        e = sp.sympify(sr.get_model_string(sr.model_, precision=6).replace("^", "**"), locals=loc)
                        if kind: e = e.xreplace({s: sp.log(s) for s in S})
                        if kind == "ll": e = sp.exp(e)
                        self.candidates.append(dict(ops=opsname, rep=rn, L=L, seed=sd, expr=e, score=score(e)))
        best = max(self.candidates, key=lambda c: c["score"]); e = refit(best["expr"], S, F[tr], y[tr])
        # prune (t02) then snap
        cur = score(e); improved = True
        while improved:
            improved = False
            for c in sorted(candidates(e), key=sp.count_ops):
                c = refit(c, S, F[tr], y[tr]); sc = score(c)
                if sc >= cur - self.tol and sp.count_ops(c) < sp.count_ops(e): e, cur, improved = c, sc, True; break
        for c in sorted(e.atoms(sp.Float), key=lambda q: -abs(float(q))):
            v = float(c); snapped = round(v * 2) / 2
            if snapped == v or abs(v) < 0.25: continue
            cand = e.xreplace({c: sp.Rational(int(snapped * 2), 2)}); cand = refit(cand, S, F[tr], y[tr]) if cand.atoms(sp.Float) else cand
            sc = score(cand)
            if sc >= cur - self.tol: e, cur = cand, max(cur, sc) if sc > cur else cur
        e = e.xreplace({c: sp.Float(float(sp.N(c, 4)), 4) for c in e.atoms(sp.Float)})
        self.choice = (best["ops"], best["rep"], best["L"], best["seed"], f"final score={cur:.4f}")
        self.formula = e; self.full_formula = e.xreplace({sp.Symbol(n): x for n, x in zip(self.pre, self.T_sym())})
        if self.logit: self.full_formula = 1 / (1 + sp.exp(-self.full_formula))
        self._f = sp.lambdify([sp.Symbol(n) for n in self.raw], self.full_formula, "numpy"); return self
