import numpy as np
def r2(y, p): return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mae(y, p): return float(np.mean(np.abs(y - p)))
def smape(y, p): return float(np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p))) * 100)
def report(y, p, prefix=""):
    return {prefix + "r2": r2(y, p), prefix + "rmse": rmse(y, p), prefix + "mae": mae(y, p), prefix + "smape": smape(y, p)}
