import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import xgboost as xgb, lightgbm as lgb

def make_teacher(name, seed=0):
    if name == "ridge": return Ridge(alpha=1.0)
    if name == "xgb": return xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, random_state=seed, n_jobs=4)
    if name == "lgbm": return lgb.LGBMRegressor(n_estimators=400, num_leaves=15, learning_rate=0.05, random_state=seed, verbose=-1, n_jobs=4)
    if name == "mlp": return MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=800, random_state=seed, early_stopping=True)
    raise ValueError(name)
