"""E08: Known closed-form laws. Teacher = MLP trained on n noisy labels. Compare symbolic regression (gplearn) fit on
  (a) the n noisy labels, (b) 5000 on-manifold teacher queries. Metrics: R2 vs noise-free truth in-range and extrapolated (x1.5 box), nodes.
Also records whether SR formula ~ truth (R2>0.999 in-range)."""
import sys, pathlib, json, warnings, time
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from mt.surrogates import gp_fit, gp_to_sympy
from mt.transforms import n_nodes
from mt.metrics import r2
OUT = ROOT / "results" / "e08"; OUT.mkdir(parents=True, exist_ok=True)
# name: (fn, [(lo,hi) per input])
LAWS = {
 "newton_grav": (lambda m1, m2, r: m1 * m2 / r**2, [(1, 5), (1, 5), (1, 5)]),
 "kinetic":     (lambda m, v: 0.5 * m * v**2, [(1, 5), (1, 5)]),
 "ideal_gas":   (lambda n, T, V: n * T / V, [(1, 5), (1, 5), (1, 5)]),
 "pendulum":    (lambda L, g: 2 * np.pi * np.sqrt(L / g), [(1, 5), (1, 5)]),
 "lens":        (lambda u, v: u * v / (u + v), [(1, 5), (1, 5)]),
 "relativistic":(lambda m, v: m / np.sqrt(1 - (v / 1.5) ** 2), [(1, 5), (0.1, 1.0)]),
 "gauss":       (lambda x, mu, s: np.exp(-((x - mu) ** 2) / (2 * s**2)) / (s * np.sqrt(2 * np.pi)), [(1, 3), (1, 3), (1, 3)]),
 "damped_osc":  (lambda A, g, t: A * np.exp(-0.3 * g * t) * np.cos(2 * t), [(1, 3), (0.2, 1), (0, 3)]),
 "snell_ratio": (lambda n1, th, n2: n1 * np.sin(th) / n2, [(1, 2), (0.1, 1.0), (1, 2)]),
}
def sample(rng, box, n, scale=1.0):
    return np.column_stack([rng.uniform(lo, lo + scale * (hi - lo), n) for lo, hi in box])
rows = []; forms = {}
for law, (fn, box) in LAWS.items():
    for seed in range(2):
        rng = np.random.default_rng(seed)
        n = 200; X = sample(rng, box, n); sig = 0.05 * np.std(fn(*X.T)); y = fn(*X.T) + rng.normal(0, sig, n)
        Xt = sample(rng, box, 2000); Xe = sample(rng, box, 2000, 1.5)
        if law == "relativistic": Xe[:, 1] = np.minimum(Xe[:, 1], 1.4)   # keep v<c
        if law == "snell_ratio": Xe[:, 1] = np.minimum(Xe[:, 1], 1.4)
        sc = StandardScaler().fit(X); Z, Zt, Ze = sc.transform(X), sc.transform(Xt), sc.transform(Xe)
        mlp = TransformedTargetRegressor(MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=4000, random_state=seed, alpha=1e-3), transformer=StandardScaler()).fit(Z, y)
        def rec(method, pt, pe, nodes=np.nan, expr=""):
            rows.append(dict(law=law, seed=seed, method=method, nodes=nodes, in_r2=r2(fn(*Xt.T), pt), ex_r2=r2(fn(*Xe.T), pe)))
            if expr: forms[f"{law}|{seed}|{method}"] = expr
        rec("mlp_teacher", mlp.predict(Zt), mlp.predict(Ze))
        Zq = sc.transform(sample(rng, box, 5000))                           # on-manifold queries (inputs independent -> uniform box)
        for src, (Zs, ys) in {"labels": (Z, y), "teacher_queries": (Zq, mlp.predict(Zq))}.items():
            t0 = time.time(); sr = gp_fit(Zs, ys, 5e-3, seed=seed, pop=1000, gens=25, funcs=("add", "sub", "mul", "div", "sqrt", "log", "sin", "cos"))
            g, s = gp_to_sympy(sr, len(box)); rec(f"gp[{src}]", sr.predict(Zt), sr.predict(Ze), n_nodes(g), s)
    print(law, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "metrics.csv", index=False); json.dump(forms, open(OUT / "formulas.json", "w"), indent=1)
pd.set_option("display.width", 200)
print(df.groupby(["law", "method"])[["nodes", "in_r2", "ex_r2"]].mean().round(3).to_string())
agg = df.groupby("method")[["in_r2", "ex_r2"]].agg(["mean", "median"]).round(3); print(agg)
