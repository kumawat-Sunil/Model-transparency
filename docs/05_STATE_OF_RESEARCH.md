# State of the research (living summary; last update after E09)

## Question
Can a trained black box (GBM / MLP) plus its preprocessing be turned into ONE explicit closed-form F(x_raw) that predicts directly from raw inputs, and when is that formula faithful, compact and useful?

## Established results (evidence → findings log id)
1. **Fusion is exact and cheap.** Imputation, log1p, scaling, one-hot, PCA, polynomial compile to sympy exactly (error 0 to 1e-15); F(raw)=g(T(raw)) equals the pipeline; ~44x faster than XGBoost inference, no ML dependency (F1, F22, F29).
2. **Not all preprocessing is closed-form.** String→code dictionaries and quantile/rank tables are lookups; exact quantile compile costs 510–10,770 nodes, dwarfing the model (F23, F29). T's cost must be part of the objective: in forecasting T was ~3.6× the size of g (F2).
3. **Compact equations often match or beat the black box they imitate, especially off-support** (F3, F14, F27, F30, F37): trees cannot extrapolate (known), MLPs extrapolate poorly, equation classes do. Fidelity-to-teacher is the wrong target off-support (F4).
4. **Distillation does not add information** (F10, F31, F35). Equal to fitting labels directly when the teacher uses the same labels; worse with a weak teacher; equal/better only when the teacher is accurate, where its role is transparency (F38). Naive augmentation hurts (F11, F16).
5. **Representation of the search space is the key lever** (F34, F39, F32): the teacher's standardization hides structure from symbolic search; raw/log spaces recover laws exactly. T_teacher ≠ T_search.
6. **Engines have complementary blind spots**: sparse-poly fails on moons/sin-abs, GP on high-dim/nonpolynomial; library mis-specification shows as node growth (F13, F28).
7. **Honest limits**: classical ETS stays competitive in forecasting (E07 correction); logistic regression equals black boxes on easy tabular tasks (F24, F25); sparse selection under non-stationarity is unreliable (F18, F32); tail risk of polynomial extrapolation (F12).

5b. **Automatic representation selection** (E10, F41-F42): choosing raw/log/z for the symbolic search by validation fidelity avoids catastrophic extrapolation failures (worst-case R2 0.14 vs -37..-462 for fixed choices) while matching the oracle median (0.926 vs 0.946).

## Candidate contributions (to test further)
- C1: Compilation-aware pipeline: joint cost over (error, fidelity, nodes(g), nodes(T)), with T_search chosen separately from T_teacher.
- C2: Verified end-to-end fusion with exactness certificate (max error on held-out raw rows) + guardrails (domain box, clipping).
- C3: Empirical map of when distillation helps (teacher accuracy, query manifold, representation) — mostly negative results that correct common assumptions.
Prior art caveat: Stage-1 and tree-extrapolation facts are known (01_LITERATURE.md); novelty unproven; a deeper literature pass is still TODO.

## Next experiments
- E10: automatic representation selection (raw/log/z per feature) with validation; ablate on E08 laws + forecasting.
- E11: joint objective optimiser + node accounting incl. T; Pareto of (error, nodes(g)+nodes(T)).
- E12: higher dimension / images (digits 3-vs-8, 64 features): scaling limits of sparse/GP.
- E13: guardrails for extrapolation tail risk; domain-of-validity certificates.
- E14: stronger SR engine (PySR if Julia obtainable; else own sparse+nonlinear-least-squares engine with exp/pow primitives).
- E15: multi-seed/CI for E07; statistical tests.
