# Model-transparency

Research: ML model (+ its preprocessing) -> explicit symbolic function -> prediction directly from raw data.

Read first: `docs/05_STATE_OF_RESEARCH.md` (summary), `docs/00_PLAN.md`, `docs/03_FINDINGS_LOG.md` (every result, incl. failures and corrections), `docs/04_RESEARCH_LOOP.md`, `docs/01_LITERATURE.md`, `docs/02_ASSUMPTIONS.md`.

Code: `src/mt/` (transforms, fuse, surrogates, compile_sklearn, data, teachers); experiments: `experiments/eNN_*.py` (each writes `results/eNN/`); tests: `python -m pytest tests`.
Requirements: numpy pandas scikit-learn scipy sympy xgboost lightgbm gplearn statsmodels matplotlib pytest. Datasets under `data/` (public: jbrownlee/Datasets, seaborn-data titanic, statsmodels CO2).
Run an experiment: `python experiments/e07_forecast_benchmark.py` (E08-E10 take 15-30 min each).
