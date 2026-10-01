"""forecasting / meantemp — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'l1': 31.3, 'l2': 31.3, 'l12': 31.3, 'month': 1.0}
RAW_HI = {'l1': 66.5, 'l2': 66.5, 'l12': 66.5, 'month': 12.0}
FEAT_LO = {'l1': 31.3, 'l2': 31.3, 'l12': 31.3, 'log_l1': 3.4436180975461075, 'log_l12': 3.4436180975461075, 'sin_m': -1.0, 'cos_m': -1.0}
FEAT_HI = {'l1': 66.5, 'l2': 66.5, 'l12': 66.5, 'log_l1': 4.197201947661808, 'log_l12': 4.197201947661808, 'sin_m': 1.0, 'cos_m': 1.0}

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
    return -2.665*log_l1*(1.288*cos_m - 0.3499*l12 + (0.6511*l1 - 0.3057*l2 - 19/2*log_l1)/(0.6149*l12 - 3.786*log_l1))/log_l12

def in_domain(l1, l2, l12, month):
    raw = dict(l1=l1, l2=l2, l12=l12, month=month); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
