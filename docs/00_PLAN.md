# Research Plan — Opening the Black Box: ML → Symbolic Function → Raw-Data Prediction

Started: 2026-10-01. Living document; updates are appended in `docs/03_FINDINGS_LOG.md`.

## 0. Operating principle (set by owner, 2026-10-01)
This is a **research** project; code is only the instrument for evaluating ideas. The goal is a defensible
conclusion (possibly a paper later). **The plan is adaptive**: after every experiment/literature round we revise
hypotheses and the next steps in `docs/04_RESEARCH_LOOP.md`. Data may be synthetic or public, whichever best answers the question.
Existing tools are used freely; the aim is to solve the problem, not to reinvent components.

## 1. Objective
Black-box models (GBMs, neural nets) are opaque. Every trained model is, mathematically, a function f(T(x_raw)).
1. **Extract** an explicit (approximate) closed-form g ≈ f.
2. **Compile** preprocessing T into the same formula so the final object is a single expression
   `F(x_raw) = g(T(x_raw))` that predicts directly from raw data, with no model object.

## 2. Research questions
- RQ1 (Fidelity vs. complexity): how close can a compact g get to teacher f, as a function of expression size?
- RQ2 (Pipeline compilation): can T + g be fused into one verifiable expression on raw inputs? Is it numerically identical to the original pipeline?
- RQ3 (Does distillation help?): is SR-on-teacher-outputs better than SR-on-labels (the control)?
- RQ4 (Forecast-specific robustness): fidelity/accuracy on future periods and under extrapolation (level shift).
- RQ5 (Method comparison): sparse-library regression vs. genetic-programming SR vs. exact tree→Piecewise.
- RQ6 (Teacher dependence): linear / GBM (XGB, LGBM) / MLP teachers — which are easier to symbolize?
- RQ7 (Scale): when does the approach break (dimensionality, interactions)? -> later phase

## 3. Honest novelty position (from literature review, see docs/01_LITERATURE.md)
Stage 1 (ML→symbolic) is established (SymTorch, GBM→SR distillation, Alaa & van der Schaar symbolic metamodels, Trepan).
Parts of Stage 2 exist (SR feature engineering, m2cgen code export). The *possibly open* piece is the
**verified end-to-end fusion** (raw → T → g as one closed-form expression) with a **joint objective over
error, model complexity AND transformation complexity** evaluated on **temporal/extrapolation** splits.
We do NOT claim novelty before a deeper search (tracked as TODO in the literature file).

## 4. Method
Joint objective: min_g  Err(g,f) [fidelity] + α·Err(g,y) [task] + λ1·C(g) + λ2·C(T_used).
Pipeline: raw → T (sympy-representable) → teacher → query set (train + perturbation samples) → surrogate search → Pareto front → substitute T → simplify → lambdify → verify vs original on raw data.

## 5. Phases
| Phase | Content | Status |
|---|---|---|
| P0 | Literature scan, plan, repo skeleton | in progress |
| P1 | Symbolic transform layer + exact compile test (preprocessing only) | |
| P2 | Synthetic forecasting benchmark with known ground truth; teachers | |
| P3 | Surrogates: sparse-library, gplearn SR, tree→Piecewise, SR-on-labels control | |
| P4 | Fusion F(x_raw)=g(T(x)) and verification; temporal + extrapolation eval | |
| P5 | Ablations: query-set augmentation, complexity penalty, teacher type | |
| P6 | Real data (public datasets reachable from sandbox), neural-net teacher (MLP) | |
| P7 | Deeper lit search, write-up | |

## 6. Experiment conventions
- Each experiment = `experiments/eNN_name.py`, writes `results/eNN_*/{metrics.json,*.csv,*.png,log.txt}`.
- Seeds fixed; all assumptions in `docs/02_ASSUMPTIONS.md`; every result + interpretation in `docs/03_FINDINGS_LOG.md` (failures included).
- Tools: sympy, gplearn (PySR needs Julia, unavailable offline → noted), scikit-learn, xgboost, lightgbm.
- Metrics: R², RMSE, MAE on y; fidelity R²/RMSE vs teacher; complexity = expression node count; inference time.
