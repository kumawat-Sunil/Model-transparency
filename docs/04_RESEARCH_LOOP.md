# Research loop (update after every round)
Format: Round N | Question | Method | Result | Conclusion | Next hypotheses

## Open hypotheses
- H1: Symbolic surrogates extrapolate beyond training range better than the tree teacher they were distilled from (trees saturate).
- H2: Distilling from a teacher (with query augmentation) beats symbolic regression on raw labels at equal complexity.
- H3: Fusing T into g is exact (to float precision) and the fused formula runs without any ML library.
- H4: A small number of terms recovers most teacher behavior when the true function is compositional; fidelity decays on interaction-heavy functions.
- H5 (to probe): the hard part is not g but T: nonsmooth/stateful preprocessing (rank/quantile transforms, one-hot, imputation, target encoding) — need an expressibility taxonomy.

## Round 0 (literature) — done, see 01_LITERATURE.md
Stage 1 established; fusion + joint complexity + temporal extrapolation = candidate gap.

## Round 1 (E01) — see findings F1–F7
Result: H1 strong yes, H3 yes, H2 no (on easy data). New: fidelity-vs-accuracy off-support is the central conceptual issue (RQ8).
Next: (E02) repair sparse engine; vary noise/n/library mis-specification to find when distillation beats label-SR (H2) and when symbolic collapses (H4);
add multi-seed. (E03) real public datasets + mis-specified truth. (E04) taxonomy of preprocessing expressibility (H5).

## Round 2-5 summary (E02-E07)
H1 supported (synthetic+real); H2 rejected for forecasting (F10,F31); H3 confirmed incl. classification+missing/one-hot (F22); H4 partly: polynomial library fails on moons/sin-abs (F28), GP rescues moons;
H5 resolved as a taxonomy: algebraic (scale, log, PCA, poly) / piecewise (impute, one-hot) / tables (quantile) with node-cost numbers (F29); H6 refuted for naive augmentation (F11,F16);
H7 (complexity growth = misspecification symptom) weakly supported (F13); H8 Pareto knee at 3-5 terms mostly holds but depends on library (F18,F32).
New: H9: teacher queries help only for small/noisy labels with abundant on-manifold queries (-> E08). H10: library prior (log, Fourier) is the real source of performance; auto-discovery of T (SR for features) is the open step.

## Round 6-7 (E08-E10)
H9 refined: distillation quality = teacher quality; teacher-queries don't beat labels with a weak teacher (F35, F38). H11 (T_teacher != T_search) supported on 19 laws across 3 experiments (F34,F39,F41); automatic selection by validation fidelity gives robustness (F42).
Next (priority): (1) SRBench/Feynman-100 style standard-benchmark run for credibility; (2) joint T+g objective optimiser (E11); (3) extrapolation guardrails/domain certificate (E13); (4) high-dim/image scaling (E12); (5) stronger SR engine / multi-seed CIs; (6) fold representation selection into the forecasting pipeline (lags: raw vs log) and re-run E07.
