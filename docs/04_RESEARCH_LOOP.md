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
