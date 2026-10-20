# Demand forecast → model → formula → prediction

Data: Kaggle "Store Item Demand Forecasting Challenge" (10 stores × 50 items, daily 2013-2017), stored in `data/` (train.csv, test.csv).

```
pip install -r requirements.txt
python stage1_train_model.py            # train LightGBM, predict          -> output/model.txt
python stage2_symbolic_regression.py    # model -> symbolic regression     -> output/formula.txt, output/formula.py
python stage3_predict_with_formula.py   # preprocess input -> formula      -> output/forecast_2018Q1.csv
```
Run from this folder, in this order.

| file | purpose |
|---|---|
| data/ | train.csv (913k rows), test.csv (45k rows) |
| data.py | loads data/train.csv and data/test.csv |
| preprocess.py | raw sales history → 5 inputs (lvl365 = last-year level ending 91 days back, lag_364, lag_728, dow, month) → features (logs, day/month flags); one definition shared by all stages |
| stage1_train_model.py | Stage 1: trains the model, shows its predictions |
| stage2_symbolic_regression.py | Stage 2: symbolic regression on the model's outputs → the formula |
| stage3_predict_with_formula.py | Stage 3: preprocesses input data and predicts with the formula only |
| output/ | generated outputs: model.txt (stage 1), formula.txt + formula.py (stage 2), forecast_2018Q1.csv (stage 3) |

## Results (2017 Q1, 45,000 rows; SMAPE, lower is better)
| | SMAPE |
|---|---|
| naive: same weekday last year | 19.15 |
| Kaggle notebook LightGBM (previous version) | 13.57 |
| **stage 1 model** (LightGBM on demand relative to yearly level) | **13.38** |
| **stage 3 formula** (found from the model in stage 2) | **13.32** |
| noise floor (Poisson noise around the true mean) | ≈ 13.25 |

The formula agrees with the model at R² 0.998 and is within ~0.07 of the best achievable error on this data.
