"""Step 4: forecast Kaggle test period (2018-01-01..2018-03-31) using ONLY formula.py (no model). Writes forecast_2018Q1.csv."""
import pathlib, numpy as np, pandas as pd
from features import load, build, RAW
import formula
HERE = pathlib.Path(__file__).resolve().parent
train, test = load("train", HERE / "data"), load("test", HERE / "data")
full = build(pd.concat([train, test.assign(sales=np.nan)], ignore_index=True))   # lags >= 91 days only use known history
fut = full[full.date >= "2018-01-01"].copy()
fut["sales"] = formula.predict(*fut[RAW].values.T)
out = test.merge(fut[["date", "store", "item", "sales"]], on=["date", "store", "item"])[["id", "date", "store", "item", "sales"]]
out.to_csv(HERE / "forecast_2018Q1.csv", index=False)
print(out.head(10).to_string(index=False)); print(f"\n{len(out)} forecasts written to forecast_2018Q1.csv; total forecast demand {out.sales.sum():,.0f}")
