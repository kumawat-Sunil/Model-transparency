"""t08 ADDITIVE SYMBOLIC DISTILLATION (different approach for many-input tabular / classification):
   g = c + sum_i f_i(x_i) [+ h(x) interaction term]
1) backfitting: each f_i is a 1-D Operon formula fit to the partial residual (3 sweeps);
2) interaction: Operon (len 20) on the remaining residual, kept only if held-out fidelity improves by >0.005;
3) joint least-squares refit of all constants, pruning (t02/t04 machinery), binary linearisation, guard, export (t05)."""
import numpy as np, sympy as sp
from pyoperon.sklearn import SymbolicRegressor
from extractors.t02_prune_refit import refit, fid
from extractors.t04_structure_rules import deep_candidates
import extractors.t05_binary_guard_export as t05
SAFE = "add,sub,mul,div,constant,variable,log,sqrt,square,exp"

def op1(x, r, L, seed):
    m = SymbolicRegressor(allowed_symbols=SAFE, generations=100, population_size=500, max_length=L, n_threads=4, random_state=seed).fit(x, r)
    return m.get_model_string(m.model_, precision=6).replace("^", "**")

class Model2Formula(t05.Model2Formula):
    def fit(self, X_raw):
        F, y = self._queries(X_raw, 0); S = [sp.Symbol(n) for n in self.pre]; k = len(S)
        rng = np.random.default_rng(1); perm = rng.permutation(len(F)); tr, ho = perm[: int(.8 * len(F))], perm[int(.8 * len(F)):]
        c0 = y[tr].mean(); parts = [sp.Integer(0)] * k; vals = np.zeros((len(F), k))
        for sweep in range(3):
            for i in range(k):
                r = y - c0 - vals.sum(1) + vals[:, i]
                if len(np.unique(F[:, i])) <= 2: e = sp.Float(float(np.polyfit(F[tr, i], r[tr], 1)[0])) * S[i]
                else:
                    s = op1(F[tr, i:i+1], r[tr], 8, sweep); e = sp.sympify(s, locals={"X1": S[i], "square": lambda q: q**2})
                f = sp.lambdify(S[i], e, "numpy")
                with np.errstate(all="ignore"): v = np.asarray(f(F[:, i]), float) * np.ones(len(F))
                if not np.all(np.isfinite(v)): continue
                v = v - v[tr].mean(); parts[i] = e - float(np.mean(np.asarray(f(F[tr, i]), float))); vals[:, i] = v
        g = sp.Float(c0) + sum(parts)
        base = fid(g, S, F[ho], y[ho])
        resid = y - (c0 + vals.sum(1))
        m = SymbolicRegressor(allowed_symbols=SAFE, generations=200, population_size=1000, max_length=20, n_threads=4, random_state=0).fit(F[tr], resid[tr])
        loc = {f"X{i+1}": S[i] for i in range(k)}; loc["square"] = lambda q: q**2
        h = sp.sympify(m.get_model_string(m.model_, precision=6).replace("^", "**"), locals=loc)
        if fid(g + h, S, F[ho], y[ho]) > base + 0.005: g = g + h
        g = refit(g, S, F[tr], y[tr]); cur = fid(g, S, F[ho], y[ho]); improved = True
        while improved:
            improved = False
            for c in sorted(deep_candidates(g), key=sp.count_ops)[:60]:
                c = refit(c, S, F[tr], y[tr]); fc = fid(c, S, F[ho], y[ho])
                if fc >= cur - 0.003 and sp.count_ops(c) < sp.count_ops(g): g, cur, improved = c, fc, True; break
        self.formula = g; self.choice = ("additive", f"held-out fidelity={cur:.4f}")
        # reuse t05 post-processing (binary linearisation, rounding, guard) by calling its body on our formula
        self.full_formula = g.xreplace({s: x for s, x in zip(S, self.T_sym())})
        if self.logit: self.full_formula = 1 / (1 + sp.exp(-self.full_formula))
        self._f = sp.lambdify([sp.Symbol(n) for n in self.raw], self.full_formula, "numpy")
        self.formula = self.formula.xreplace({c: sp.Float(float(sp.N(c, 4)), 4) for c in self.formula.atoms(sp.Float)})
        self.full_formula = self.formula.xreplace({s: x for s, x in zip(S, self.T_sym())})
        if self.logit: self.full_formula = 1 / (1 + sp.exp(-self.full_formula))
        self._f = sp.lambdify([sp.Symbol(n) for n in self.raw], self.full_formula, "numpy")
        self.raw_lo, self.raw_hi = X_raw.min(0), X_raw.max(0); Ft = self.T(X_raw); self.f_lo, self.f_hi = Ft.min(0), Ft.max(0)
        return self
