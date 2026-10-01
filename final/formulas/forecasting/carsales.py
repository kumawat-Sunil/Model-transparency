"""forecasting / carsales — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'l1': 5568.0, 'l2': 5568.0, 'l12': 5568.0, 'month': 1.0}
RAW_HI = {'l1': 23541.0, 'l2': 23541.0, 'l12': 23541.0, 'month': 12.0}
FEAT_LO = {'l1': 5568.0, 'l2': 5568.0, 'l12': 5568.0, 'log_l1': 8.624791202014256, 'log_l12': 8.624791202014256, 'sin_m': -1.0, 'cos_m': -1.0}
FEAT_HI = {'l1': 23541.0, 'l2': 23541.0, 'l12': 23541.0, 'log_l1': 10.066498860795583, 'log_l12': 10.066498860795583, 'sin_m': 1.0, 'cos_m': 1.0}

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
    return -2763/2*log_l1 + 31.2*np.sqrt(13.75*l1 + 2.898*l12*sin_m + 41.7*l12 + 3.097*l2)

def in_domain(l1, l2, l12, month):
    raw = dict(l1=l1, l2=l2, l12=l12, month=month); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
