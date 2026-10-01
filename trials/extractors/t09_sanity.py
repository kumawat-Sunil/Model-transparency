"""t09 numerical-sanity filter for candidate formulas g (in feature symbols), evaluated on query features F:
  reject if (a) any |numeric exponent| > 6, (b) any constant |c| > 1e8 or 0 < |c| < 1e-10,
            (c) cancellation: max std of an additive term > 20 x std of the whole output, (d) non-finite outputs."""
import numpy as np, sympy as sp
def sane(g, S, F):
    for p in g.atoms(sp.Pow):
        if p.exp.is_number and abs(float(p.exp)) > 6: return False, "exponent>6"
    for c in g.atoms(sp.Float):
        v = abs(float(c))
        if v > 1e8 or (0 < v < 1e-10): return False, "extreme constant"
    with np.errstate(all="ignore"):
        tot = np.asarray(sp.lambdify(S, g, "numpy")(*F.T), float) * np.ones(len(F))
        if not np.all(np.isfinite(tot)): return False, "non-finite"
        if isinstance(g, sp.Add):
            for t in g.args:
                v = np.asarray(sp.lambdify(S, t, "numpy")(*F.T), float) * np.ones(len(F))
                if np.std(v) > 20 * (np.std(tot) + 1e-12): return False, "cancelling terms"
    return True, "ok"
