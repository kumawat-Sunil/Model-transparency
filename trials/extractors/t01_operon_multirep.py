"""model2formula: trained ML model + its preprocessing  ->  one formula  F(raw) = g(T(raw)).
Usage:
    m2f = Model2Formula(model, preprocess_exprs, raw_names).fit(X_raw_train)
    m2f.formula            # g in terms of preprocessed features
    m2f.full_formula       # fused: in terms of RAW inputs only
    m2f.predict(X_raw)     # prediction using ONLY the formula (no model)
preprocess_exprs: dict feature_name -> function(env, L) using L.log/L.sin/... (works for numpy AND sympy, so T is exact)."""
import numpy as np, sympy as sp
from pyoperon.sklearn import SymbolicRegressor
from mt.transforms import _SymLib

OPS = "add,sub,mul,div,constant,variable,sin,cos,log,exp,sqrt,square"


class Model2Formula:
    def __init__(self, model, preprocess_exprs, raw_names, model_input="features", max_length=25, n_query=5000, seed=0, logit=False):
        self.model, self.pre, self.raw, self.model_input = model, preprocess_exprs, list(raw_names), model_input
        self.max_length, self.n_query, self.seed, self.logit = max_length, n_query, seed, logit

    # ---- T: numeric and symbolic twins of the same preprocessing ----
    def T(self, X):
        env = {n: X[:, i] for i, n in enumerate(self.raw)}
        return np.column_stack([np.asarray(f(env, np), float) * np.ones(len(X)) for f in self.pre.values()])

    def T_sym(self):
        env = {n: sp.Symbol(n) for n in self.raw}
        return [f(env, _SymLib) for f in self.pre.values()]

    def _model_out(self, X):
        inp = self.T(X) if self.model_input == "features" else X
        if self.logit:
            p = np.clip(self.model.predict_proba(inp)[:, 1], 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
        return self.model.predict(inp)

    def fit(self, X_raw):
        rng = np.random.default_rng(self.seed)
        # queries: training rows + rows resampled per-column inside the observed data (stays on realistic values)
        idx = rng.integers(0, len(X_raw), (self.n_query, X_raw.shape[1]))
        Xq = np.vstack([X_raw, X_raw[idx, np.arange(X_raw.shape[1])]])
        Fq, yq = self.T(Xq), self._model_out(Xq)
        ok = np.all(np.isfinite(Fq), 1) & np.isfinite(yq); Fq, yq = Fq[ok], yq[ok]
        names = list(self.pre); S = [sp.Symbol(n) for n in names]
        cut = int(0.8 * len(Fq)); perm = rng.permutation(len(Fq)); a, b = perm[:cut], perm[cut:]   # held-out queries for selection
        reps = {"raw": (lambda F: F, lambda e: e)}
        if np.all(Fq > 0): reps["log-inputs"] = (np.log, lambda e: e.xreplace({s_: sp.log(s_) for s_ in S}))
        if np.all(Fq > 0) and np.all(yq > 0): reps["log-log"] = (np.log, "loglog")
        self.candidates = []
        for rn, (fx, back) in reps.items():
            ty = np.log(yq) if back == "loglog" else yq
            for L in sorted({10, 20, self.max_length}):
                sr = SymbolicRegressor(allowed_symbols=OPS, generations=200, population_size=1000, max_length=L, n_threads=4, random_state=self.seed)
                sr.fit(fx(Fq[a]), ty[a])
                loc = {f"X{i+1}": S[i] for i in range(len(S))}; loc["square"] = lambda q: q ** 2
                e = sp.sympify(sr.get_model_string(sr.model_, precision=6).replace("^", "**"), locals=loc)
                e = sp.exp(e.xreplace({s_: sp.log(s_) for s_ in S})) if back == "loglog" else back(e)
                f = sp.lambdify(S, e, "numpy")
                with np.errstate(all="ignore"): pb = np.asarray(f(*Fq[b].T), float) * np.ones(len(b))
                fid = 1 - np.nanmean((pb - yq[b]) ** 2) / np.var(yq[b]) if np.all(np.isfinite(pb)) else -np.inf
                nodes = sum(1 for _ in sp.preorder_traversal(e))
                self.candidates.append(dict(rep=rn, max_len=L, fidelity=fid, nodes=nodes, score=fid - 0.0005 * nodes, expr=e))
        best = max(self.candidates, key=lambda c: c["score"]); self.choice = (best["rep"], best["max_len"], best["fidelity"])
        self.formula = best["expr"]
        self.full_formula = self.formula.xreplace({sp.Symbol(n): e for n, e in zip(names, self.T_sym())})
        if self.logit: self.full_formula = 1 / (1 + sp.exp(-self.full_formula))
        self._f = sp.lambdify([sp.Symbol(n) for n in self.raw], self.full_formula, "numpy")
        return self

    def predict(self, X_raw):  # FORMULA ONLY
        return np.asarray(self._f(*X_raw.T), float) * np.ones(len(X_raw))

    def model_predict(self, X_raw):
        out = self._model_out(X_raw); return 1 / (1 + np.exp(-out)) if self.logit else out
