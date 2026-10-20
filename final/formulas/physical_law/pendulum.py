"""physical_law / pendulum — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'L': 1.0003175544718466, 'g': 1.0027449647045024}
RAW_HI = {'L': 4.993353001622695, 'g': 4.99115814434103}
FEAT_LO = {'L': 1.0003175544718466, 'g': 1.0027449647045024}
FEAT_HI = {'L': 4.993353001622695, 'g': 4.99115814434103}

def features(L, g):
    """Preprocessing T (raw -> model features)."""
    L = L
    g = g
    return dict(L=L, g=g)

def predict(L, g):
    f = features(L, g); L = f['L']; g = f['g']
    return (13/2)*L**0.4737*g**(-0.5015)

def in_domain(L, g):
    raw = dict(L=L, g=g); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
