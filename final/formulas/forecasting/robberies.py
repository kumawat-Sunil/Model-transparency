"""forecasting / robberies — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'l1': 32.0, 'l2': 29.0, 'l12': 29.0, 'month': 1.0}
RAW_HI = {'l1': 401.0, 'l2': 401.0, 'l12': 353.0, 'month': 12.0}
FEAT_LO = {'l1': 32.0, 'l2': 29.0, 'l12': 29.0, 'log_l1': 3.4657359027997265, 'log_l12': 3.367295829986474, 'sin_m': -1.0, 'cos_m': -1.0}
FEAT_HI = {'l1': 401.0, 'l2': 401.0, 'l12': 353.0, 'log_l1': 5.993961427306569, 'log_l12': 5.8664680569332965, 'sin_m': 1.0, 'cos_m': 1.0}

def features(l1, l2, l12, month):
    """Preprocessing T (raw -> model features)."""
    l1 = l1
    l2 = l2
    l12 = l12
    log_l1 = np.log(l1)
    log_l12 = np.log(l12)
    sin_m = np.sin((1/6)*np.pi*month)
    cos_m = np.cos((1/6)*np.pi*month)
    return dict(l1=l1, l2=l2, l12=l12, log_l1=log_l1, log_l12=log_l12, sin_m=sin_m, cos_m=cos_m)

def predict(l1, l2, l12, month):
    f = features(l1, l2, l12, month); l1 = f['l1']; l2 = f['l2']; l12 = f['l12']; log_l1 = f['log_l1']; log_l12 = f['log_l12']; sin_m = f['sin_m']; cos_m = f['cos_m']
    return (241/2)*np.log(-0.001958*log_l12*(-1.344*l1 - 34.67*l1/l2 - 1.432*l2))

def in_domain(l1, l2, l12, month):
    raw = dict(l1=l1, l2=l2, l12=l12, month=month); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
