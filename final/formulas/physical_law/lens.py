"""physical_law / lens — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'u': 1.0007075090773845, 'v': 1.003510867714577}
RAW_HI = {'u': 4.9998594728088905, 'v': 4.997060435618348}
FEAT_LO = {'u': 1.0007075090773845, 'v': 1.003510867714577}
FEAT_HI = {'u': 4.9998594728088905, 'v': 4.997060435618348}

def features(u, v):
    """Preprocessing T (raw -> model features)."""
    u = u
    v = v
    return dict(u=u, v=v)

def predict(u, v):
    f = features(u, v); u = f['u']; v = f['v']
    return (1/2)*u**0.414*v**0.4155*np.exp(0.1192*np.log(u)*np.log(v))

def in_domain(u, v):
    raw = dict(u=u, v=v); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
