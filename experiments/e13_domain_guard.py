"""E13 (raw-box flag OR engineered-feature (z) box flag). Original: E13: does a raw-input training-box certificate predict symbolic extrapolation failures?
Re-runs the E02 misspec/inlib regimes (noise 2/8, nser 6/15, 8 seeds). Sparse surrogate <- XGB. Rows flagged if any raw input outside [min,max] of train.
Reports error (vs noise-free truth) inside vs outside box, flag rate, and tail (max/p95) error with and without flagged rows."""
import sys, pathlib, warnings, itertools; ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from mt.data import make_series, make_pipeline, true_mean, true_mean_misspec
from mt.teachers import make_teacher
from mt.surrogates import sparse_select
from mt.fuse import to_callable, eval_on_raw
rows = []
for truth, noise, nser, seed in itertools.product(["inlib", "misspec"], [2.0, 8.0], [6, 15], range(8)):
    fn = true_mean if truth == "inlib" else true_mean_misspec
    d = make_series(nser, 96, seed=seed, fn=fn, noise=noise); ex = make_series(20, 96, seed=1000 + seed, base_range=(60, 200), fn=fn, noise=noise)
    tr = d["t"] < 60; va = (d["t"] >= 60) & (d["t"] < 78); Xtr = d["X"][tr]; lo, hi = Xtr.min(0), Xtr.max(0)
    pipe = make_pipeline().fit(Xtr); k = len(pipe.names); Ztr, Zva = pipe.transform(Xtr), pipe.transform(d["X"][va])
    xg = make_teacher("xgb", seed).fit(Ztr, d["y"][tr])
    g, _ = sparse_select(Ztr, xg.predict(Ztr), Zva, d["y"][va]); f = to_callable(g, [f"z{i}" for i in range(k)])
    Xe = ex["X"]; p = eval_on_raw(f, pipe.transform(Xe)); err = np.abs(p - ex["mu"]); Ze = pipe.transform(Xe); zlo, zhi = Ztr.min(0), Ztr.max(0); outz = ((Ze < zlo) | (Ze > zhi)).any(1); outr = ((Xe < lo) | (Xe > hi)).any(1); out = outr | outz
    # price & month never leave box in practice; flag on level inputs only is captured in 'out'
    rows.append(dict(truth=truth, noise=noise, nser=nser, seed=seed, flag_rate=out.mean(), flag_raw_only=outr.mean(), flag_z_only=outz.mean(), mae_in=err[~out].mean() if (~out).any() else np.nan,
                     mae_out=err[out].mean() if out.any() else np.nan, p95_all=np.percentile(err, 95), p95_unflagged=np.percentile(err[~out], 95) if (~out).any() else np.nan,
                     max_all=err.max(), max_unflagged=err[~out].max() if (~out).any() else np.nan, teacher_mae_out=np.abs(xg.predict(pipe.transform(Xe))-ex["mu"])[out].mean() if out.any() else np.nan))
df = pd.DataFrame(rows); (ROOT / "results/e13").mkdir(exist_ok=True); df.to_csv(ROOT / "results/e13/metrics.csv", index=False)
pd.set_option("display.width", 200); print(df.groupby(["truth", "noise", "nser"]).median(numeric_only=True).drop(columns="seed").round(2).to_string())
print("\noverall medians:\n", df.median(numeric_only=True).round(2).to_string()); print("\nworst max_all", df.max_all.max().round(1), "worst max_unflagged", df.max_unflagged.max().round(1))
