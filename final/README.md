# FINAL — ML model → formula → prediction from raw data

One call: `extract(model, preprocess, raw_names, X_train_raw, problem_type, model_input, classification)` (model2formula.py)
returns a formula F(raw) = g(T(raw)) that replaces the model, a trust-region check `in_domain`, and `export()` to a numpy-only .py file.

## Best method per problem type (each verified on 3–5 datasets)
| problem type | method | what it does |
|---|---|---|
| physical / smooth law | t06 Operon search | multi-representation (raw/log/log-log) symbolic search on model queries, fidelity-first selection, prune + least-squares refit + constant snapping |
| time-series forecasting | t06 + periodic rule | same; trend features get no sin/cos (seasonality comes from month features) -> formulas extrapolate trends |
| tabular regression | t09 sparse polynomial (selected among t06 / t08 / t09) | lasso path over features, squares, pairwise products of the preprocessed features; smallest support within 0.005 of best held-out fidelity; numerical sanity filter |
| binary classification | t09 sparse polynomial on the logit | same, output = 1/(1+exp(-g)) |

## Verification (results/FINAL_RESULTS.csv)
| type | dataset | model | formula | fidelity R² (formula vs model) | nodes | note |
|---|---|---|---|---|---|---|
| physical | gravity | R² 0.997 | **1.000** | 0.997 | 6 | recovered exactly: `m1*m2/r**2` (extrapolation R² 1.000) |
| physical | pendulum | 0.996 | 0.999 | 0.995 | 8 | `6.5*L**0.474/g**0.502` (truth 2π√(L/g)) |
| physical | lens | 0.996 | 0.991 | 0.993 | 15 | |
| physical | rc_decay | 0.995 | 0.997 | 0.992 | 11 | |
| forecasting | airline | 0.380 | **0.840** | 0.443* | 8 | `0.206*l1 + 0.757*l12 + 30` |
| forecasting | carsales | 0.664 | 0.741 | 0.872 | 27 | |
| forecasting | robberies | -0.588 | -0.796 | -0.065 | 20 | model itself fails on this test window |
| forecasting | meantemp | 0.881 | 0.921 | 0.970 | 41 | |
| forecasting | co2 | 0.585 | **0.918** | -0.202* | 7 | `0.165*l1 + 0.839*l12` |
| tabular | diabetes | 0.431 | 0.426 | 0.962 | 68 | |
| tabular | boston | 0.895 | 0.854 | 0.981 | 179 | |
| tabular | auto_mpg (one-hot origin) | 0.889 | 0.872 | 0.993 | 110 | |
| classification (AUC) | titanic | 0.886 | 0.884 | 0.970 | 78 | |
| classification | pima | 0.755 | 0.789 | 0.868 | 173 | |
| classification | breast_cancer | 0.992 | 0.993 | 0.997 | 127 | |
| classification | penguins | 1.000 | 1.000 | 0.993 | 45 | |
\* forecasting test months lie above the training range: the tree model cannot extrapolate, the formula can, so they disagree there by design (in_domain flags those rows).

Every exported file in `formulas/<type>/<dataset>.py` reproduces the in-memory formula (max diff ≤ 1e-5).

## Files
- `model2formula.py` (current), `model2formula_v1.py` (previous selection without sanity filter / sparse candidate)
- `engine/` — methods (copied from trials t01–t09)
- `verify.py` — re-runs the verification; `results/` — csv + all formulas (`verification_v2_formulas.txt` for tabular/classification, `verification_formulas.txt` for physical/forecasting)
- `formulas/` — standalone formula files per dataset

## Known limits
Tabular/classification formulas are faithful (fidelity 0.87–0.997) but long (45–179 nodes, mostly pairwise products); physical/forecasting formulas are short. Forecasting fidelity is only meaningful inside the training range.
