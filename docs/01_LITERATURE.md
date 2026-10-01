# Literature notes (web scan 2026-10-01; titles/claims as returned by search, NOT yet read in full — verify before citing)

## Stage 1: ML → symbolic
- SymTorch (arXiv 2602.21307): wraps NN components, collects I/O, fits equations with PySR; used to cut transformer MLP compute.
- Follow the Forest Trail (IEEE 11363421): GBM teacher → SR student, +3.3–6.7% accuracy vs plain SR. **Closest to our "teacher helps SR" claim (RQ3).**
- Distilling RL policies: GBM + SR (arXiv 2403.14328).
- NestyNet-SR III (2608.21051); Recovery-directed symbolic distillation of neural likelihoods (2609.32409); KAN-based GNN→symbolic (2512.20885).
- Closed-form interpretation of NN with symbolic gradients (2401.04978, 2409.05305).
- Alaa & van der Schaar 2019, symbolic metamodels (Meijer G-functions).
- Classical rule/tree extraction: Trepan (Craven & Shavlik 1995), Born-Again Trees (Breiman & Shang 1996); tree-ensemble compression (2206.07904); model-based trees as surrogates (2310.03112).

## Stage 2: transformations / deployment
- SR as feature engineering (2311.06028); SymboLLM-FE (2608.28408) — SR formulas → executable FE code.
- m2cgen: transpiles trained sklearn/XGB/LGBM models (incl. pipelines) to dependency-free code. **Exact conversion exists** but yields huge code, not a compact interpretable formula. Relevant baseline for "exact but opaque-size".
- sklearn-onnx: pipeline → graph.

## Forecasting
- SRLinear (SR features for long-term TS forecasting); symbolic forecasting w/ neural+evolutionary search; SR for chaotic time series (2603.07261); Towards transparent time series forecasting (ICLR 2024).

## Gaps / TODO (deeper search needed)
1. Any work that fuses preprocessing symbolic maps with distilled g and verifies raw→prediction equivalence?
2. Joint complexity metric over T and g.
3. Temporal-extrapolation behavior of distilled symbolic forecasters (trees cannot extrapolate; equations can — hypothesis H1).
4. Sources to read in full: Follow the Forest Trail, SymTorch, SRLinear, m2cgen docs.

## Round 2 (2026-10-01, snippets only; verify)
- Tree-model extrapolation failure is textbook knowledge (leaf averages cannot exceed training range; Snowflake blog "Comparing transform techniques for tree-based models", SETAR-Tree arXiv 2211.08661, many practitioner posts). => our F3/F14 "symbolic extrapolates, trees don't" is an expected effect, NOT a discovery; value is only in quantifying it within a distillation+fusion pipeline.
- Feynman SR benchmarks: AI Feynman 2.0, "Rethinking SR datasets and benchmarks" (2206.10540), PhySO, ParFam, MOSAIC-SR (2609.20997), neural-guided equation discovery (2503.16953); "Interpretability in SR: benchmark of explanatory methods on Feynman" (2404.05908). Standard practice: AI-Feynman uses NN as a teacher to discover separability/symmetries; log-transform to find power laws is classical (Kepler-style).
- No source found specifically on "input standardization hurts SR"; E08/E08b test it empirically.
- Symbolic ML for chaotic time series (2603.07261).

## Round 3 (2026-10-01, snippets only; verify)
- Searches for "symbolic distillation + preprocessing fused into one raw-input closed form with equivalence check" returned only Stage-1 frameworks: SymTorch (2602.21307; wraps NN components, PySR), KAN-SR (2509.10089), SMILE (2609.04639; prune -> parametric opt -> coefficient rounding), Recovery-directed distillation (2609.32409), Neural Symbolic Regression w/ sparse modeling (2609.01102), "NN as functional preconditioner" (neural smoothing + sparse regression). **No hit for fused T+g verification or a joint T+g complexity objective** (absence in snippets, not proof).
- AI Feynman (Udrescu & Tegmark, Sci. Adv.): NN used as smooth oracle to detect separability/symmetry; teacher-as-oracle is established (consistent with F37).
- SR-community awareness that normalization can corrupt symbolic form (MOSAIC-SR 2609.20997; pretrained SR works that omit affine skeleton transforms): our F34 (z-scoring hurts) is an instance of a known concern; the novel part would be the explicit T_teacher / T_search split plus automatic representation selection (E10).
- SRBench / SRBench++ (252 problems incl. Feynman) is the right standard benchmark to run before any paper claim.
