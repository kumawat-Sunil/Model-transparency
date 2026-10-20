"""Verifies the FINAL method on 16 datasets (4 problem types x 3-5 datasets). Writes final/results/verification.csv, final/results/formulas.txt,
and one standalone formula file per dataset in final/formulas/<type>/<dataset>.py (checked to reproduce the in-memory formula)."""
import sys, pathlib, importlib.util, warnings; HERE = pathlib.Path(__file__).parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "trials")); sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp, time
from sklearn.metrics import r2_score, roc_auc_score
from model2formula import extract
import bench2, os
ONLY = os.environ.get("ONLY_TYPES")
rows = []; txt = []
for tk in bench2.tasks():
    if ONLY and tk["type"] not in ONLY.split(","): continue
    t0 = time.time()
    res = extract(tk["model"], tk["pre"], tk["raw"], tk["Xtr"], tk["type"], model_input=tk["model_input"], classification=tk["logit"])
    pf, pm = np.nan_to_num(res.predict(tk["Xte"])), res.model_predict(tk["Xte"]); sc = r2_score if tk["kind"] == "reg" else roc_auc_score
    d = HERE / "formulas" / tk["type"]; d.mkdir(parents=True, exist_ok=True); fp = d / f"{tk['name']}.py"; res.export(fp, f"{tk['type']} / {tk['name']}")
    spec = importlib.util.spec_from_file_location("m", fp); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    ok = float(np.max(np.abs(np.asarray(m.predict(**{n: tk["Xte"][:, i] for i, n in enumerate(tk["raw"])}), float) * np.ones(len(tk["Xte"])) - res.predict(tk["Xte"]))))
    r = dict(type=tk["type"], dataset=tk["name"], method=res.method, candidates=str(getattr(res, "validation", "")), metric="R2" if tk["kind"] == "reg" else "AUC", model_score=round(sc(tk["yte"], pm), 4), formula_score=round(sc(tk["yte"], pf), 4),
             fidelity_R2=round(r2_score(pm, pf), 4), nodes=sum(1 for _ in sp.preorder_traversal(res.full_formula)), in_domain_test=round(float(np.mean(res.in_domain(tk["Xte"]))), 3),
             export_max_diff=ok, secs=round(time.time() - t0, 1))
    if "Xex" in tk: r["extrapolation_R2"] = round(r2_score(tk["yex"], np.nan_to_num(res.predict(tk["Xex"]))), 4)
    rows.append(r); pd.DataFrame(rows).to_csv(HERE / "results" / (os.environ.get("TAG", "verification") + ".csv"), index=False)
    txt.append(f"## {tk['type']} / {tk['name']}  [{res.method}]\npreprocessing: { {k: str(e) for k, e in zip(tk['pre'], res.T_sym())} }\ny = {sp.N(res.full_formula, 5)}\n{r}\n")
    (HERE / "results" / (os.environ.get("TAG", "verification") + "_formulas.txt")).write_text("\n".join(txt)); print(txt[-1], flush=True)
