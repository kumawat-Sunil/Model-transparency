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


def true_mean_misspec(l1, l2, l3, l12, price, month, lib=np):
    """Truth OUTSIDE the deg-2 polynomial library: saturating demand, price ratio, product of seasonal and exp."""
    roll = (l1 + l2 + l3) / 3
    seas = 1 + 0.25 * lib.sin(2 * lib.pi * month / 12)
    return (0.5 * l1 / (1 + lib.abs(l1) / 300) + 0.25 * l12 * seas + 0.15 * roll
            - 25 * lib.log(price / 10) + 8 * lib.sqrt(lib.abs(l2 - l3) + 1) * lib.sin(price / 3) + 5)


def make_series(n_series=40, T=96, seed=0, level_shift_after=None, shift=1.0, base_range=(40, 120), fn=None, noise=2.0):
    fn = fn or true_mean
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
            mu = fn(l1, l2, l3, l12, price_t, month)
            yt = mu + rng.normal(0, noise)
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


# ---------- real univariate series: raw = lag window + calendar month ----------
RAW_REAL = ["l1", "l2", "l3", "l6", "l12", "month"]


def real_feature_exprs():
    lg = lambda x, L: L.log(1 + x)
    return {
        "lg1": lambda e, L: lg(e["l1"], L), "lg2": lambda e, L: lg(e["l2"], L), "lg3": lambda e, L: lg(e["l3"], L),
        "lg6": lambda e, L: lg(e["l6"], L), "lg12": lambda e, L: lg(e["l12"], L),
        "lgroll3": lambda e, L: lg((e["l1"] + e["l2"] + e["l3"]) / 3, L),
        "sin_m": lambda e, L: L.sin(2 * L.pi * e["month"] / 12), "cos_m": lambda e, L: L.cos(2 * L.pi * e["month"] / 12),
        "sin2_m": lambda e, L: L.sin(4 * L.pi * e["month"] / 12), "cos2_m": lambda e, L: L.cos(4 * L.pi * e["month"] / 12),
    }


def make_real_pipeline():
    return SymbolicPipeline(RAW_REAL, real_feature_exprs())


def window_xy(y, months):
    """Rows for t>=12: raw=[y[t-1],y[t-2],y[t-3],y[t-6],y[t-12],month[t]], target y[t]."""
    y = np.asarray(y, float); idx = np.arange(12, len(y))
    X = np.column_stack([y[idx - 1], y[idx - 2], y[idx - 3], y[idx - 6], y[idx - 12], months[idx]])
    return idx, X, y[idx]


def real_feature_exprs_raw():
    """Ablation: no log transforms (levels only) + Fourier month."""
    return {
        "l1": lambda e, L: e["l1"], "l2": lambda e, L: e["l2"], "l3": lambda e, L: e["l3"], "l6": lambda e, L: e["l6"], "l12": lambda e, L: e["l12"],
        "roll3": lambda e, L: (e["l1"] + e["l2"] + e["l3"]) / 3,
        "sin_m": lambda e, L: L.sin(2 * L.pi * e["month"] / 12), "cos_m": lambda e, L: L.cos(2 * L.pi * e["month"] / 12),
        "sin2_m": lambda e, L: L.sin(4 * L.pi * e["month"] / 12), "cos2_m": lambda e, L: L.cos(4 * L.pi * e["month"] / 12),
    }


def real_feature_exprs_nofourier():
    """Ablation: log lags but month only as raw number (no Fourier terms)."""
    f = real_feature_exprs(); [f.pop(k) for k in ("sin_m", "cos_m", "sin2_m", "cos2_m")]
    f["month"] = lambda e, L: e["month"]; return f


def make_real_pipeline_kind(kind="full"):
    ex = {"full": real_feature_exprs, "nolog": real_feature_exprs_raw, "nofourier": real_feature_exprs_nofourier}[kind]()
    return SymbolicPipeline(RAW_REAL, ex)
