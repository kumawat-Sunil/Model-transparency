"""Synthetic demand-forecasting panel with KNOWN generating function."""
import numpy as np
from .transforms import SymbolicPipeline

RAW = ["l1", "l2", "l3", "l12", "price", "month"]


def true_mean(l1, l2, l3, l12, price, month, lib=np):
    """Ground truth E[y_t | history]. Nonlinear: log-price elasticity, seasonal, momentum x seasonality interaction."""
    roll = (l1 + l2 + l3) / 3
    seas = 1 + 0.25 * lib.sin(2 * lib.pi * month / 12) + 0.1 * lib.cos(4 * lib.pi * month / 12)
    return (0.45 * l1 + 0.2 * roll + 0.25 * l12 * seas
            - 18 * lib.log(price / 10) + 0.002 * (l1 - l12) * (l1 - roll) + 5)


def make_series(n_series=40, T=96, seed=0, level_shift_after=None, shift=1.0, base_range=(40, 120)):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_series):
        base = rng.uniform(*base_range)
        price = rng.uniform(8, 14)
        y = list(base * (1 + 0.1 * np.sin(2 * np.pi * np.arange(12) / 12)) + rng.normal(0, 2, 12))
        for t in range(12, T):
            month = (t % 12) + 1
            price_t = price * (1 + 0.15 * np.sin(t / 7 + s)) * (1 + rng.normal(0, 0.02))
            l1, l2, l3, l12 = y[t - 1], y[t - 2], y[t - 3], y[t - 12]
            mu = true_mean(l1, l2, l3, l12, price_t, month)
            yt = mu + rng.normal(0, 2.0)
            if level_shift_after is not None and t >= level_shift_after:
                yt = yt * shift  # exogenous regime shift -> forces extrapolation
            y.append(yt)
            rows.append((s, t, l1, l2, l3, l12, price_t, month, yt, mu))
    a = np.array(rows)
    return dict(series=a[:, 0].astype(int), t=a[:, 1].astype(int), X=a[:, 2:8], y=a[:, 8], mu=a[:, 9])


def feature_exprs():
    return {
        "l1": lambda e, L: e["l1"],
        "l2": lambda e, L: e["l2"],
        "l3": lambda e, L: e["l3"],
        "l12": lambda e, L: e["l12"],
        "roll3": lambda e, L: (e["l1"] + e["l2"] + e["l3"]) / 3,
        "logp": lambda e, L: L.log(e["price"]),
        "sin_m": lambda e, L: L.sin(2 * L.pi * e["month"] / 12),
        "cos_m": lambda e, L: L.cos(2 * L.pi * e["month"] / 12),
        "sin2_m": lambda e, L: L.sin(4 * L.pi * e["month"] / 12),
        "cos2_m": lambda e, L: L.cos(4 * L.pi * e["month"] / 12),
        "mom": lambda e, L: (e["l1"] - e["l12"]) / (e["l12"]),
    }


def make_pipeline():
    return SymbolicPipeline(RAW, feature_exprs())
