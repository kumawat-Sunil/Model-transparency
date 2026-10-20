"""t04 = t03 + (1) STRUCTURE RULE: if the preprocessing already supplies periodic features (sin/cos in T), the search uses the SAFE
operator set only (no sin/cos/exp applied to trend features -> no oscillation off-range); (2) DEEP pruning: drop Add-terms at any depth."""
import numpy as np, sympy as sp
from engine.t02_prune_refit import candidates as shallow
import engine.t03_snap_safe as t03

def deep_candidates(e):
    out = shallow(e)
    for sub in sp.preorder_traversal(e):
        if isinstance(sub, sp.Add) and sub is not e:
            for t in sub.args: out.append(e.xreplace({sub: sub - t}))
    return out
t03.candidates = deep_candidates  # used by t03's prune loop

class Model2Formula(t03.Model2Formula):
    def fit(self, X_raw):
        periodic = any(x.has(sp.sin) or x.has(sp.cos) for x in self.T_sym())
        if periodic:
            orig = t03.t01.OPS; t03.t01.OPS = t03.SAFE
            try: super().fit(X_raw)
            finally: t03.t01.OPS = orig
            self.choice = (*self.choice, "periodic-rule: SAFE ops only")
            return self
        return super().fit(X_raw)
