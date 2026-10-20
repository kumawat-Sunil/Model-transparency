"""E01: teacher -> symbolic surrogate -> fused raw-input formula, on synthetic demand forecasting.
Evaluates: in-time test (future periods), extrapolation (higher-level series), fidelity, complexity, fusion exactness."""
import sys, pathlib, json, time, warnings
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp
from mt.data import make_series, make_pipeline, RAW, true_mean
from mt.teachers import make_teacher
from mt.surrogates import sparse_fit, gp_fit, gp_to_sympy, tree_node_count
from mt.fuse import fuse, to_callable, eval_on_raw
from mt.metrics import report, r2
from mt.transforms import n_nodes

OUT = ROOT / "results" / "e01"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
d = make_series(40, 96, seed=SEED)
tr = d["t"] < 60; va = (d["t"] >= 60) & (d["t"] < 78); te = d["t"] >= 78
ex = make_series(20, 96, seed=123, base_range=(130, 220))   # extrapolation: levels unseen in training
pipe = make_pipeline().fit(d["X"][tr])
Z = {k: pipe.transform(m) for k, m in dict(tr=d["X"][tr], va=d["X"][va], te=d["X"][te], ex=ex["X"]).items()}
Y = dict(tr=d["y"][tr], va=d["y"][va], te=d["y"][te], ex=ex["y"])
RAWX = dict(tr=d["X"][tr], te=d["X"][te], ex=ex["X"])
zexprs = pipe.z_exprs(); k = Z["tr"].shape[1]
rows = []; formulas = {}

# oracle: true generating function (upper bound on achievable accuracy)
rows.append(dict(teacher="oracle", method="truth", complexity=np.nan, **report(Y["te"], true_mean(*RAWX["te"].T), "te_"),
                 **report(Y["ex"], true_mean(*RAWX["ex"].T), "ex_")))

rng = np.random.default_rng(SEED)
def augment(Zt, n=6000, scale=0.3):
    idx = rng.integers(0, len(Zt), n)
    return Zt[idx] + rng.normal(0, scale, (n, Zt.shape[1])) * Zt.std(0)

for tname in ["ridge", "xgb", "lgbm", "mlp"]:
    t = make_teacher(tname, SEED).fit(Z["tr"], Y["tr"])
    P = {s: t.predict(Z[s]) for s in Z}
    base = dict(teacher=tname)
    tc = None
    if tname in ("xgb", "lgbm"): tc = tree_node_count(t, tname)
    rows.append(dict(**base, method="teacher", complexity=tc, **report(Y["te"], P["te"], "te_"), **report(Y["ex"], P["ex"], "ex_"),
                     fid_te_r2=1.0, fid_ex_r2=1.0))
    # distillation query sets
    for qname, Zq in [("train", Z["tr"]), ("train+aug", np.vstack([Z["tr"], augment(Z["tr"])]))]:
        yq = t.predict(Zq)
        # A: sparse library
        for alpha in [1e-1, 1e-2, 1e-3]:
            t0 = time.time(); g, nterm = sparse_fit(Zq, yq, alpha)
            fn = to_callable(g, [f"z{i}" for i in range(k)])
            pg = {s: eval_on_raw(fn, Z[s]) for s in ("te", "ex")}
            rows.append(dict(**base, method=f"sparse[{qname}]", alpha=alpha, complexity=n_nodes(g), terms=nterm,
                             **report(Y["te"], pg["te"], "te_"), **report(Y["ex"], pg["ex"], "ex_"),
                             fid_te_r2=r2(P["te"], pg["te"]), fid_ex_r2=r2(P["ex"], pg["ex"]), secs=time.time() - t0))
            formulas[f"{tname}|sparse[{qname}]|{alpha}"] = str(g)
        # B: GP symbolic regression
        for pc in [1e-3, 1e-2]:
            t0 = time.time()
            sub = rng.choice(len(Zq), min(3000, len(Zq)), replace=False)
            sr = gp_fit(Zq[sub], yq[sub], pc, seed=SEED, pop=800, gens=20)
            g, raw_s = gp_to_sympy(sr, k)
            pg = {s: sr.predict(Z[s]) for s in ("te", "ex")}
            rows.append(dict(**base, method=f"gp[{qname}]", parsimony=pc, complexity=n_nodes(g), 
                             **report(Y["te"], pg["te"], "te_"), **report(Y["ex"], pg["ex"], "ex_"),
                             fid_te_r2=r2(P["te"], pg["te"]), fid_ex_r2=r2(P["ex"], pg["ex"]), secs=time.time() - t0))
            formulas[f"{tname}|gp[{qname}]|{pc}"] = raw_s
        print(tname, qname, "done", flush=True)

# control: SR/sparse directly on labels (no teacher)
for alpha in [1e-1, 1e-2, 1e-3]:
    g, nterm = sparse_fit(Z["tr"], Y["tr"], alpha)
    fn = to_callable(g, [f"z{i}" for i in range(k)])
    pg = {s: eval_on_raw(fn, Z[s]) for s in ("te", "ex")}
    rows.append(dict(teacher="none(labels)", method="sparse[labels]", alpha=alpha, complexity=n_nodes(g), terms=nterm,
                     **report(Y["te"], pg["te"], "te_"), **report(Y["ex"], pg["ex"], "ex_")))
for pc in [1e-3, 1e-2]:
    sr = gp_fit(Z["tr"][:3000], Y["tr"][:3000], pc, seed=SEED, pop=800, gens=20); g, raw_s = gp_to_sympy(sr, k)
    pg = {s: sr.predict(Z[s]) for s in ("te", "ex")}
    rows.append(dict(teacher="none(labels)", method="gp[labels]", parsimony=pc, complexity=n_nodes(g),
                     **report(Y["te"], pg["te"], "te_"), **report(Y["ex"], pg["ex"], "ex_")))
    formulas[f"labels|gp|{pc}"] = raw_s

# ---- fusion check on the best sparse surrogate of the xgb teacher: F(raw) == g(T(raw)) ----
t = make_teacher("xgb", SEED).fit(Z["tr"], Y["tr"]); Zq = np.vstack([Z["tr"], augment(Z["tr"])])
g, _ = sparse_fit(Zq, t.predict(Zq), 1e-2)
F = fuse(g, zexprs); fF = to_callable(F, RAW)
gfn = to_callable(g, [f"z{i}" for i in range(k)])
err = float(np.max(np.abs(eval_on_raw(fF, RAWX["te"]) - eval_on_raw(gfn, Z["te"]))))
t0 = time.time(); _ = eval_on_raw(fF, RAWX["te"]); tF = time.time() - t0
t0 = time.time(); _ = t.predict(Z["te"]); tM = time.time() - t0
fusion = dict(max_abs_diff_fused_vs_composed=err, fused_nodes=n_nodes(F), g_nodes=n_nodes(g), n_rows=int(len(RAWX["te"])),
              fused_secs=tF, xgb_secs=tM, fused_formula=str(sp.N(F, 5)))
json.dump(fusion, open(OUT / "fusion.json", "w"), indent=1)
json.dump(formulas, open(OUT / "formulas.json", "w"), indent=1)
df = pd.DataFrame(rows); df.to_csv(OUT / "metrics.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
print(df[["teacher", "method", "complexity", "te_r2", "te_rmse", "ex_r2", "ex_rmse", "fid_te_r2", "fid_ex_r2"]].round(3).to_string())
print(fusion)
