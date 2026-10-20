"""Step 1-3: (1) train the notebook-style LightGBM demand model, (2) symbolic regression (Operon) on the model's predictions,
(3) write formula.py (standalone) + results.md.   Run:  python train_and_extract.py"""
import pathlib, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp, lightgbm as lgb
from pyoperon.sklearn import SymbolicRegressor
from features import load, build, RAW, PRE, T, smape
HERE = pathlib.Path(__file__).resolve().parent
t0 = time.time()

# ---------- 1. data + LightGBM (validation = 2017 Q1, like the notebook) ----------
df = build(load("train", HERE / "data")).dropna(subset=RAW)
tr = df[df.date < "2017-01-01"]; va = df[(df.date >= "2017-01-01") & (df.date < "2017-04-01")]
Xtr, Xva = T(tr[RAW].values), T(va[RAW].values)
params = dict(objective="regression", metric="mae", num_leaves=10, learning_rate=0.02, feature_fraction=0.8, verbose=-1, num_threads=4)
model = lgb.train(params, lgb.Dataset(Xtr, np.log1p(tr.sales.values)), 3000, valid_sets=[lgb.Dataset(Xva, np.log1p(va.sales.values))],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
p_lgb = np.expm1(model.predict(Xva))
print(f"LightGBM: {model.best_iteration} trees, validation SMAPE {smape(va.sales.values, p_lgb):.3f}")

# ---------- 2. symbolic regression on the MODEL's output (log scale) ----------
rng = np.random.default_rng(0)
fit_idx, sel_idx = rng.choice(len(tr), 20000, replace=False), rng.choice(len(tr), 20000, replace=False)
Ffit, Fsel = Xtr[fit_idx], Xtr[sel_idx]; yfit, ysel = model.predict(Ffit), model.predict(Fsel)
S = [sp.Symbol(n) for n in PRE]; cands = []
for L in (10, 20, 35, 50):
    for seed in (0, 1):
        sr = SymbolicRegressor(allowed_symbols="add,sub,mul,div,constant,variable,square", generations=300, population_size=1500,
                               max_length=L, n_threads=4, random_state=seed).fit(Ffit, yfit)
        e = sp.sympify(sr.get_model_string(sr.model_, precision=6).replace("^", "**"), locals={**{f"X{i+1}": s for i, s in enumerate(S)}, "square": lambda q: q**2})
        fid = 1 - np.mean((sr.predict(Fsel) - ysel) ** 2) / np.var(ysel)
        cands.append((fid - 0.0002 * sum(1 for _ in sp.preorder_traversal(e)), fid, L, seed, e)); print(f"  len {L} seed {seed}: fidelity {fid:.4f}")
_, fid, L, seed, g = max(cands, key=lambda c: c[0])
# linear polish: keep the SR terms, add every calendar feature linearly, least-squares on the model output
cal = [sp.Symbol(n) for n in PRE if n.startswith(("dow_", "sin_month", "cos_month"))]
terms = [t for t in (g.args if isinstance(g, sp.Add) else [g]) if not t.is_number]
basis = list(dict.fromkeys([t.as_coeff_Mul()[1] for t in terms] + cal))
B = lambda F: np.column_stack([np.asarray(sp.lambdify(S, b, "numpy")(*F.T), float) * np.ones(len(F)) for b in basis] + [np.ones(len(F))])
coef, *_ = np.linalg.lstsq(B(Ffit), yfit, rcond=None)
g2 = sum(sp.Float(float(c), 5) * b for c, b in zip(coef[:-1], basis)) + sp.Float(float(coef[-1]), 5)
fid2 = 1 - np.mean((B(Fsel) @ coef - ysel) ** 2) / np.var(ysel); print(f"  linear polish: fidelity {fid:.4f} -> {fid2:.4f}")
if fid2 > fid: g, fid = g2, fid2
g = g.xreplace({c: sp.Float(float(sp.N(c, 5)), 5) for c in g.atoms(sp.Float)})

# ---------- 3. fuse preprocessing into the formula, evaluate, export ----------
from sympy import log, sin, cos, pi
class SymL: log, sin, cos, pi = staticmethod(log), staticmethod(sin), staticmethod(cos), pi
raw_sym = {n: sp.Symbol(n) for n in RAW}
feat_exprs = {n: f(raw_sym, SymL) for n, f in PRE.items()}
log_sales = g                                                     # formula in features
sales = sp.expm1(log_sales) if hasattr(sp, "expm1") else sp.exp(log_sales) - 1
f_fn = sp.lambdify([raw_sym[n] for n in RAW], sp.exp(g.xreplace({sp.Symbol(n): e for n, e in feat_exprs.items()})) - 1, "numpy")
p_formula = f_fn(*va[RAW].values.T)
res = dict(lgb_smape=smape(va.sales.values, p_lgb), formula_smape=smape(va.sales.values, p_formula), naive_lag364_smape=smape(va.sales.values, va.lag_364.values),
           fidelity_log_R2=1 - np.mean((np.log1p(p_formula) - np.log1p(p_lgb)) ** 2) / np.var(np.log1p(p_lgb)), nodes=sum(1 for _ in sp.preorder_traversal(g)))
feat_code = "\n".join(f"    {n} = {sp.pycode(e).replace('math.', 'np.')}" for n, e in feat_exprs.items())
(HERE / "formula.py").write_text(f'''"""Demand formula extracted from the LightGBM model by symbolic regression (numpy only).
predict(lag_91, lag_98, lag_182, lag_364, lag_728, ewm95_lag91, ewm50_lag91, ewm50_lag364, roll365_lag91, dow, month) -> daily sales"""
import numpy as np

def predict({", ".join(RAW)}):
{feat_code}
    log_sales = {sp.pycode(g).replace("math.", "np.")}
    return np.expm1(log_sales)
''')
txt = f"""# Results (validation = 2017-01-01..2017-03-31, 500 store-item series, {len(va)} rows)

| model | SMAPE |
|---|---|
| seasonal naive (sales 364 days ago) | {res['naive_lag364_smape']:.3f} |
| LightGBM (notebook recipe, {model.best_iteration} trees) | {res['lgb_smape']:.3f} |
| **extracted formula** | **{res['formula_smape']:.3f}** |

Formula vs LightGBM agreement (R² on log scale): {res['fidelity_log_R2']:.4f}; formula size: {res['nodes']} nodes (chosen: max_length {L}, seed {seed}).

## Formula
log(1 + sales) = {sp.N(g, 4)}

where log_X = log(1 + X) of each raw input, dow_d = 1 if day_of_week == d (Mon = 0 is base), sin_month/cos_month = sin/cos(2π·month/12).
sales = exp(log(1 + sales)) − 1.  Standalone code: `formula.py`.  Runtime: {time.time() - t0:.0f}s
"""
(HERE / "results.md").write_text(txt); print(txt)
