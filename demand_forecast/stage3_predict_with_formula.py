"""STAGE 3 — use the formula on new input data: raw sales history -> preprocess -> formula -> prediction (no model).
Checks 2017 Q1 against actual sales and the model, then forecasts the Kaggle test period 2018 Q1. Output: forecast_2018Q1.csv"""
import numpy as np, pandas as pd, lightgbm as lgb
from data import load_train, load_test
from preprocess import make_inputs, features, INPUTS, smape
import formula

train, test = load_train(), load_test()
# 1) preprocess input data: build the formula's inputs from raw history (test rows have no sales yet)
rows = make_inputs(pd.concat([train, test.assign(sales=np.nan)], ignore_index=True))

# 2) check on 2017 Q1 (known sales): formula vs model vs actual
va = rows[(rows.date >= "2017-01-01") & (rows.date < "2017-04-01")]
p_formula = formula.predict(*va[INPUTS].values.T)
p_model = np.expm1(np.log1p(va.lvl365.values) + lgb.Booster(model_file="model.txt").predict(features(va[INPUTS].values)))
print(f"2017 Q1 SMAPE  formula {smape(va.sales.values, p_formula):.2f} | model {smape(va.sales.values, p_model):.2f}")
print(va.assign(formula=p_formula.round(1), model=p_model.round(1))[["date", "store", "item", "sales", "model", "formula"]].head(7).to_string(index=False))

# 3) forecast 2018 Q1 with the formula only
fut = rows[rows.date >= "2018-01-01"].copy(); fut["sales"] = formula.predict(*fut[INPUTS].values.T)
out = test.merge(fut[["date", "store", "item", "sales"]], on=["date", "store", "item"])[["id", "date", "store", "item", "sales"]]
out.to_csv("forecast_2018Q1.csv", index=False)
print(f"\n2018 Q1 forecast with the formula: {len(out)} rows -> forecast_2018Q1.csv (total demand {out.sales.sum():,.0f})")
