"""t05 = t04 + (1) binary-variable linearisation: any sub-expression depending only on a 0/1 raw variable b (and constants)
is replaced by f(0) + (f(1)-f(0))*b, then constants refit; (2) domain guard: training min/max of raw inputs AND of g's features;
(3) export(): writes a standalone .py (numpy only) with preprocessing, formula, guard."""
import numpy as np, sympy as sp, textwrap
import engine.t04_structure_rules as t04
from engine.t02_prune_refit import refit, fid

class Model2Formula(t04.Model2Formula):
    def fit(self, X_raw):
        super().fit(X_raw)
        S = [sp.Symbol(n) for n in self.pre]; F, y = self._queries(X_raw, 0)
        binary = [s for s, col in zip(S, F.T) if set(np.unique(col)) <= {0.0, 1.0}]
        e = self.formula
        for b in binary:
            for sub in sorted(sp.preorder_traversal(e), key=sp.count_ops, reverse=True):
                if sub.is_Atom or sub.free_symbols != {b} or sub.is_polynomial(b) and sp.degree(sub, b) <= 1: continue
                lin = sub.subs(b, 0) + (sub.subs(b, 1) - sub.subs(b, 0)) * b
                e = e.xreplace({sub: sp.nsimplify(lin) if False else sp.N(lin, 6)})
            e = e.subs(b**2, b)
        if e != self.formula:
            e2 = refit(e, S, F, y)
            if fid(e2, S, F, y) >= fid(self.formula, S, F, y) - 0.003: self.formula = e2; self.choice = (*self.choice, f"binary-linearised {binary}")
        self.formula = self.formula.xreplace({c: sp.Float(float(sp.N(c, 4)), 4) for c in self.formula.atoms(sp.Float)})
        self.full_formula = self.formula.xreplace({sp.Symbol(n): x for n, x in zip(self.pre, self.T_sym())})
        if self.logit: self.full_formula = 1 / (1 + sp.exp(-self.full_formula))
        self._f = sp.lambdify([sp.Symbol(n) for n in self.raw], self.full_formula, "numpy")
        self.raw_lo, self.raw_hi = X_raw.min(0), X_raw.max(0); Ft = self.T(X_raw); self.f_lo, self.f_hi = Ft.min(0), Ft.max(0)
        return self

    def in_domain(self, X_raw):
        F = self.T(X_raw); return ~(np.any((X_raw < self.raw_lo) | (X_raw > self.raw_hi), 1) | np.any((F < self.f_lo) | (F > self.f_hi), 1))

    def export(self, path, title):
        feats = "\n".join(f"    {n} = {sp.pycode(x).replace('math.', 'np.')}" for n, x in zip(self.pre, self.T_sym()))
        g = sp.pycode(self.formula).replace("math.", "np.")
        if self.logit: g = f"1 / (1 + np.exp(-({g})))"
        src = f'''"""{title} — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {dict(zip(self.raw, map(float, self.raw_lo)))}
RAW_HI = {dict(zip(self.raw, map(float, self.raw_hi)))}
FEAT_LO = {dict(zip(self.pre, map(float, self.f_lo)))}
FEAT_HI = {dict(zip(self.pre, map(float, self.f_hi)))}

def features({", ".join(self.raw)}):
    """Preprocessing T (raw -> model features)."""
{feats}
    return dict({", ".join(f"{n}={n}" for n in self.pre)})

def predict({", ".join(self.raw)}):
    f = features({", ".join(self.raw)}); {"; ".join(f"{n} = f['{n}']" for n in self.pre)}
    return {g}

def in_domain({", ".join(self.raw)}):
    raw = dict({", ".join(f"{n}={n}" for n in self.raw)}); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
'''
        open(path, "w").write(src)
