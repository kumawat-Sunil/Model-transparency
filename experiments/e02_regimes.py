"""E02: when does distillation beat label-only symbolic fitting, and when does the symbolic route break?
Sweeps truth (in-library / mis-specified), noise sd, #series; 5 seeds; repaired sparse engine; XGB teacher."""
import sys, pathlib, json, warnings, itertools
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from mt.data import make_series, make_pipeline, true_mean, true_mean_misspec
from mt.teachers import make_teacher
from mt.surrogates import sparse_select
from mt.fuse import to_callable, eval_on_raw
from mt.metrics import r2, rmse
from mt.transforms import n_nodes

OUT = ROOT / "results" / "e02"; OUT.mkdir(parents=True, exist_ok=True)
rows = []
for truth, noise, nser, seed in itertools.product(["inlib", "misspec"], [2.0, 8.0, 20.0], [6, 15, 40], range(5)):
    fn = true_mean if truth == "inlib" else true_mean_misspec
    d = make_series(nser, 96, seed=seed, fn=fn, noise=noise)
    ex = make_series(20, 96, seed=1000 + seed, base_range=(130, 220), fn=fn, noise=noise)
    tr = d["t"] < 60; va = (d["t"] >= 60) & (d["t"] < 78); te = d["t"] >= 78
    pipe = make_pipeline().fit(d["X"][tr]); k = len(pipe.names)
    Z = {n: pipe.transform(m) for n, m in dict(tr=d["X"][tr], va=d["X"][va], te=d["X"][te], ex=ex["X"]).items()}
    Y = dict(tr=d["y"][tr], va=d["y"][va], te=d["y"][te], ex=ex["y"])
    MU = dict(te=d["mu"][te], ex=ex["mu"])
    xg = make_teacher("xgb", seed).fit(Z["tr"], Y["tr"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(Z["tr"]), 6000)
    Zaug = np.vstack([Z["tr"], Z["tr"][idx] + rng.normal(0, 0.3, (6000, k)) * Z["tr"].std(0)])
    cand = {"labels": (Z["tr"], Y["tr"]), "distill": (Z["tr"], xg.predict(Z["tr"])), "distill+aug": (Zaug, xg.predict(Zaug))}
    res = {"xgb_teacher": (None, xg.predict)}
    for name, (Zq, yq) in cand.items():
        g, nt = sparse_select(Zq, yq, Z["va"], Y["va"])
        fn_ = to_callable(g, [f"z{i}" for i in range(k)])
        res[name] = (n_nodes(g), lambda Zs, f=fn_: eval_on_raw(f, Zs))
    for name, (cx, pred) in res.items():
        r = dict(truth=truth, noise=noise, nser=nser, seed=seed, method=name, complexity=cx)
        for s in ("te", "ex"):
            p = pred(Z[s]); r[f"{s}_rmse_vs_truth"] = rmse(MU[s], p); r[f"{s}_r2_y"] = r2(Y[s], p)
        rows.append(r)
    print(truth, noise, nser, seed, flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "metrics.csv", index=False)
agg = df.groupby(["truth", "noise", "nser", "method"])[["complexity", "te_rmse_vs_truth", "ex_rmse_vs_truth"]].median().round(2)
agg.to_csv(OUT / "summary_median.csv"); pd.set_option("display.max_rows", 300); pd.set_option("display.width", 200)
print(agg.to_string())
