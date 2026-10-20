"""E11: joint objective (error, fused nodes incl. T) over E07 variants. Pure re-analysis of results/e07/metrics.csv."""
import pathlib, pandas as pd, numpy as np
R = pathlib.Path(__file__).parents[1] / "results"; df = pd.read_csv(R / "e07/metrics.csv"); df = df[df.method.str.startswith("sym")]
rows = []
for s, g in df.groupby("series"):
    g = g.copy(); g["pareto"] = [not ((g.rec12_smape < r.rec12_smape) & (g.nodes <= r.nodes)).any() and not ((g.rec12_smape <= r.rec12_smape) & (g.nodes < r.nodes)).any() for r in g.itertuples()]
    best = g.loc[g.rec12_smape.idxmin()]; cheap = g[g.rec12_smape <= best.rec12_smape * 1.10].sort_values("nodes").iloc[0]
    rows.append(dict(series=s, best_method=best.method, best_smape=best.rec12_smape, best_nodes=best.nodes, within10pct_method=cheap.method, within10pct_smape=cheap.rec12_smape, within10pct_nodes=cheap.nodes,
                     pareto_methods=";".join(g[g.pareto].method)))
o = pd.DataFrame(rows); o.to_csv(R / "e11_joint_pareto.csv", index=False); pd.set_option("display.width", 220); print(o.round(2).to_string())
print("median nodes best:", o.best_nodes.median(), "| median nodes within 10% of best:", o.within10pct_nodes.median())
