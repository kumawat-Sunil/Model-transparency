"""t09 candidate: SPARSE POLYNOMIAL distillation (always numerically clean): lasso path over {features, squares, pairwise products},
support size chosen on held-out teacher queries (smallest support within 0.005 of best fidelity), ridge refit. Uses t05 guard/export."""
import itertools, numpy as np, sympy as sp
from sklearn.linear_model import lasso_path, Ridge
import extractors.t05_binary_guard_export as t05

class Model2Formula(t05.Model2Formula):
    def fit(self, X_raw):
        F, y = self._queries(X_raw, 0); S = [sp.Symbol(n) for n in self.pre]; k = len(S)
        terms = [(S[i], F[:, i]) for i in range(k)] + [(S[i] * S[j], F[:, i] * F[:, j]) for i, j in itertools.combinations_with_replacement(range(k), 2)]
        A = np.column_stack([t[1] for t in terms]); keep = A.std(0) > 1e-12; A = A[:, keep]; terms = [t for t, kp in zip(terms, keep) if kp]
        mu, sd = A.mean(0), A.std(0); As = (A - mu) / sd
        rng = np.random.default_rng(1); perm = rng.permutation(len(F)); tr, ho = perm[: int(.8 * len(F))], perm[int(.8 * len(F)):]
        _, coefs, _ = lasso_path(As[tr], y[tr] - y[tr].mean(), n_alphas=60, eps=1e-4)
        path = []
        for j in range(coefs.shape[1]):
            sup = tuple(np.flatnonzero(np.abs(coefs[:, j]) > 1e-10))
            if not sup: continue
            m = Ridge(alpha=1e-6).fit(As[tr][:, sup], y[tr]); p = m.predict(As[ho][:, sup]); path.append((len(sup), 1 - np.mean((p - y[ho]) ** 2) / np.var(y[ho]), sup, m))
        best = max(p[1] for p in path); n, f_, sup, m = min([p for p in path if p[1] >= best - 0.005], key=lambda p: p[0])
        g = sp.Float(float(m.intercept_ - np.sum(m.coef_ * mu[list(sup)] / sd[list(sup)])), 6)
        for c, i in zip(m.coef_, sup): g += sp.Float(float(c / sd[i]), 5) * terms[i][0]
        self.formula = g; self.choice = ("sparse-poly", n, f"held-out fidelity={f_:.4f}")
        self.full_formula = g.xreplace({s: x for s, x in zip(S, self.T_sym())})
        if self.logit: self.full_formula = 1 / (1 + sp.exp(-self.full_formula))
        self._f = sp.lambdify([sp.Symbol(n_) for n_ in self.raw], self.full_formula, "numpy")
        self.raw_lo, self.raw_hi = X_raw.min(0), X_raw.max(0); Ft = self.T(X_raw); self.f_lo, self.f_hi = Ft.min(0), Ft.max(0)
        return self
