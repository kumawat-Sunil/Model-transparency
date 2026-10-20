"""Shared feature building (same recipe as the Kaggle LightGBM notebook: lags >= 91 days, lagged EWM / rolling means, calendar).
Raw formula inputs are these columns; the formula itself applies log1p / one-hot internally."""
import numpy as np, pandas as pd
URL = "https://raw.githubusercontent.com/ugursaricam/store-item-demand-forecasting/master/datasets/{}.csv"  # mirror of Kaggle "Store Item Demand Forecasting Challenge"

RAW = ["lag_91", "lag_98", "lag_182", "lag_364", "lag_728", "ewm95_lag91", "ewm50_lag91", "ewm50_lag364", "roll365_lag91", "dow", "month"]

def load(name, data_dir):
    p = data_dir / f"{name}.csv"
    if not p.exists():
        data_dir.mkdir(exist_ok=True); pd.read_csv(URL.format(name)).to_csv(p, index=False)
    return pd.read_csv(p, parse_dates=["date"])

def build(df):
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    g = df.groupby(["store", "item"])["sales"]
    for L in (91, 98, 182, 364, 728): df[f"lag_{L}"] = g.shift(L)
    df["ewm95_lag91"] = g.transform(lambda x: x.shift(91).ewm(alpha=0.95).mean())
    df["ewm50_lag91"] = g.transform(lambda x: x.shift(91).ewm(alpha=0.5).mean())
    df["ewm50_lag364"] = g.transform(lambda x: x.shift(364).ewm(alpha=0.5).mean())
    df["roll365_lag91"] = g.transform(lambda x: x.shift(91).rolling(365, min_periods=300).mean())
    df["dow"] = df.date.dt.dayofweek.astype(float); df["month"] = df.date.dt.month.astype(float)
    return df

# preprocessing T: raw inputs -> model features. Written once, works with numpy (L=np) and sympy (L=sympy facade).
PRE = {f"log_{c}": (lambda e, L, c=c: L.log(1 + e[c])) for c in RAW[:9]}
for d in range(1, 7):  # one-hot day of week (Mon = base), exact Lagrange indicator on integer codes 0..6
    def ind(e, L, d=d):
        v = 1
        for o in range(7):
            if o != d: v = v * (e["dow"] - o) / (d - o)
        return v
    PRE[f"dow_{d}"] = ind
PRE["sin_month"] = lambda e, L: L.sin(2 * L.pi * e["month"] / 12)
PRE["cos_month"] = lambda e, L: L.cos(2 * L.pi * e["month"] / 12)

def T(X):
    env = {n: X[:, i] for i, n in enumerate(RAW)}
    return np.column_stack([np.asarray(f(env, np), float) * np.ones(len(X)) for f in PRE.values()])

def smape(y, p): return float(100 * np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p) + 1e-9)))
