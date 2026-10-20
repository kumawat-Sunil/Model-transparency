"""forecasting / co2 — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'l1': 313.4, 'l2': 313.4, 'l12': 313.4, 'month': 1.0}
RAW_HI = {'l1': 370.975, 'l2': 370.975, 'l12': 369.14, 'month': 12.0}
FEAT_LO = {'l1': 313.4, 'l2': 313.4, 'l12': 313.4, 'log_l1': 5.74748032992192, 'log_l12': 5.74748032992192, 'sin_m': -1.0, 'cos_m': -1.0}
FEAT_HI = {'l1': 370.975, 'l2': 370.975, 'l12': 369.14, 'log_l1': 5.916134674892191, 'log_l12': 5.911175975879146, 'sin_m': 1.0, 'cos_m': 1.0}

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
    return 0.1646*l1 + 0.8395*l12

def in_domain(l1, l2, l12, month):
    raw = dict(l1=l1, l2=l2, l12=l12, month=month); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
