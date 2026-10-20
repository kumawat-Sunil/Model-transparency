"""E10: automatic selection of the SYMBOLIC SEARCH representation (z / raw / log) by validation on a mildly extended box.
10 new laws incl. exp primitives; strong MLP teacher (n=3000); GP on 5000 teacher queries per representation.
Selection score = R2(vs teacher, validation box 1.3x) - 0.002*nodes.  Reports selected vs oracle-best vs fixed choices."""
import sys, pathlib, json, warnings
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from mt.surrogates import gp_fit, gp_to_sympy
from mt.transforms import n_nodes
from mt.metrics import r2
OUT = ROOT / "results" / "e10"; OUT.mkdir(parents=True, exist_ok=True)
LAWS = {
 "coulomb":   (lambda q1, q2, r: q1 * q2 / r**2, [(1, 5), (1, 5), (1, 5)], True),
 "wave_speed":(lambda T, mu: np.sqrt(T / mu), [(1, 5), (1, 5)], True),
 "doppler":   (lambda f, c, v: f * c / (c + v), [(1, 5), (3, 6), (0.5, 2)], True),
 "rc_decay":  (lambda V, t, R, C: V * np.exp(-t / (R * C)), [(1, 5), (0.1, 2), (1, 3), (0.5, 2)], True),
 "planck":    (lambda x, T: x**3 / (np.exp(x / T) - 1), [(0.5, 3), (1, 3)], True),
 "logistic":  (lambda K, t, r: K / (1 + np.exp(-r * (t - 2))), [(1, 5), (0, 4), (0.5, 2)], True),
 "thin_film": (lambda n, d, th: n * d * np.cos(th), [(1, 2), (1, 5), (0.1, 1.2)], True),
 "drag":      (lambda rho, v, Cd, A: 0.5 * rho * v**2 * Cd * A, [(1, 3), (1, 5), (0.5, 1.5), (1, 3)], True),
 "heat":      (lambda m, c, T1, T2: m * c * (T2 - T1), [(1, 5), (1, 3), (1, 3), (3, 6)], True),
 "range":     (lambda v, th, g: v**2 * np.sin(2 * th) / g, [(1, 5), (0.1, 1.4), (1, 3)], True),
}
def sample(rng, box, n, s=1.0): return np.column_stack([rng.uniform(lo, lo + s * (hi - lo), n) for lo, hi, *_ in [(b[0], b[1]) for b in box]])
FUNCS = ("add", "sub", "mul", "div", "sqrt", "log", "exp", "sin", "cos")
rows = []
for law, (fn, box, _) in LAWS.items():
    rng = np.random.default_rng(0)
    X = sample(rng, box, 3000); y = fn(*X.T) + rng.normal(0, 0.01 * np.std(fn(*X.T)), len(X))
    sc = StandardScaler().fit(X)
    mlp = TransformedTargetRegressor(MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=4000, random_state=0, alpha=1e-3), transformer=StandardScaler()).fit(sc.transform(X), y)
    T = lambda A: mlp.predict(sc.transform(A))
    Xt, Xv, Xe = sample(rng, box, 2000), sample(rng, box, 2000, 1.3), sample(rng, box, 2000, 1.8)
    Xq = sample(rng, box, 5000); yq = T(Xq)
    reps = {"z": (lambda A: sc.transform(A)), "raw": (lambda A: A), "log": (lambda A: np.log(A))}
    res = {}
    for rn, f in reps.items():
        tgt = yq if rn != "log" else np.log(np.clip(yq, 1e-6, None))
        sr = gp_fit(f(Xq), tgt, 2e-3, pop=1500, gens=30, funcs=FUNCS); g, s_ = gp_to_sympy(sr, len(box))
        back = (lambda A, sr=sr, f=f: sr.predict(f(A))) if rn != "log" else (lambda A, sr=sr, f=f: np.exp(np.clip(sr.predict(f(A)), -50, 50)))
        nodes = n_nodes(g)
        res[rn] = dict(nodes=nodes, in_r2=r2(fn(*Xt.T), back(Xt)), ex_r2=r2(fn(*Xe.T), back(Xe)), val_fid=r2(T(Xv), back(Xv)), expr=s_)
        res[rn]["score"] = res[rn]["val_fid"] - 0.002 * nodes
        rows.append(dict(law=law, rep=rn, **{k: v for k, v in res[rn].items()}))
    sel = max(res, key=lambda r: res[r]["score"]); orc = max(res, key=lambda r: res[r]["ex_r2"])
    rows.append(dict(law=law, rep="SELECTED:" + sel, **{k: res[sel][k] for k in ("nodes", "in_r2", "ex_r2", "val_fid", "score")}, expr=res[sel]["expr"]))
    rows.append(dict(law=law, rep="ORACLE:" + orc, **{k: res[orc][k] for k in ("nodes", "in_r2", "ex_r2", "val_fid", "score")}, expr=res[orc]["expr"]))
    print(law, "sel", sel, "oracle", orc, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
df = pd.DataFrame(rows); pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
print(df[["law", "rep", "nodes", "in_r2", "ex_r2", "val_fid"]].round(3).to_string())
df["kind"] = df.rep.str.split(":").str[0]; print(df[df.kind.isin(["z", "raw", "log", "SELECTED", "ORACLE"])].groupby("kind")[["in_r2", "ex_r2", "nodes"]].median().round(3))
