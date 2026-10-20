"""Preprocessing — ONE definition used by the model (stage 1), the symbolic regression (stage 2) and the formula (stage 3).
  make_inputs(df): raw daily sales history -> inputs per row:
      lvl365  = mean sales of the 364 days ending 91 days before the date (recent yearly level; 91 = forecast horizon)
      lag_364, lag_728 = sales on the same weekday 1 and 2 years earlier;  dow (0=Mon..6=Sun);  month (1..12)
  features(X): inputs -> model features (FEATURE_CODE below, numpy expressions; the same text is written into formula.py)"""
import numpy as np

INPUTS = ["lvl365", "lag_364", "lag_728", "dow", "month"]
FEATURE_CODE = {"log_lvl365": "np.log1p(lvl365)", "log_lag_364": "np.log1p(lag_364)", "log_lag_728": "np.log1p(lag_728)",
                **{f"dow_{d}": f"(dow == {d}) * 1.0" for d in range(1, 7)},
                **{f"month_{m}": f"(month == {m}) * 1.0" for m in range(2, 13)}}
FEATURES = list(FEATURE_CODE)

def make_inputs(df):
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    s = df.groupby(["store", "item"])["sales"]
    df["lvl365"] = s.transform(lambda x: x.shift(91).rolling(364, min_periods=364).mean())
    df["lag_364"], df["lag_728"] = s.shift(364), s.shift(728)
    df["dow"], df["month"] = df.date.dt.dayofweek.astype(float), df.date.dt.month.astype(float)
    return df

def features(X):
    env = {"np": np, **{n: X[:, i] for i, n in enumerate(INPUTS)}}
    return np.column_stack([np.asarray(eval(code, env), float) * np.ones(len(X)) for code in FEATURE_CODE.values()])

def smape(y, p): return float(100 * np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p) + 1e-9)))
