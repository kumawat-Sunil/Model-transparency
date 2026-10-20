"""STAGE 1 — train the demand model and predict with it.
LightGBM learns  r = log(1+sales) - log(1+lvl365)  (demand relative to the recent yearly level) from the features.
Prediction: sales = exp(log(1+lvl365) + r) - 1.  Train: up to 2016, validate: 2017 Q1.  Output: output/model.txt"""
import pathlib
OUT = pathlib.Path(__file__).resolve().parent / "output"; OUT.mkdir(exist_ok=True)
import numpy as np, lightgbm as lgb
from data import load_train
from preprocess import make_inputs, features, INPUTS, smape

df = make_inputs(load_train()).dropna(subset=INPUTS)
tr, va = df[df.date < "2017-01-01"], df[(df.date >= "2017-01-01") & (df.date < "2017-04-01")]
Ftr, Fva = features(tr[INPUTS].values), features(va[INPUTS].values)
r_tr = np.log1p(tr.sales.values) - np.log1p(tr.lvl365.values)
model = lgb.train(dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, verbose=-1, num_threads=4),
                  lgb.Dataset(Ftr, r_tr), num_boost_round=1500)
model.save_model(str(OUT / "model.txt"))

pred = np.expm1(np.log1p(va.lvl365.values) + model.predict(Fva))
print(f"model trained: {model.num_trees()} trees -> output/model.txt")
print(f"2017 Q1 SMAPE: model {smape(va.sales.values, pred):.2f} | naive 'same weekday last year' {smape(va.sales.values, va.lag_364.values):.2f}")
print(va.assign(model_prediction=pred.round(1))[["date", "store", "item", "sales", "model_prediction"]].head(7).to_string(index=False))
