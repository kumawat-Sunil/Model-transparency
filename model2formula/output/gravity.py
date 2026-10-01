"""gravity — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'m1': 1.000760006429374, 'm2': 1.0150481135348186, 'r': 1.0012027604276916}
RAW_HI = {'m1': 4.995768570369693, 'm2': 4.998228995967311, 'r': 4.998005409028107}
FEAT_LO = {'m1': 1.000760006429374, 'm2': 1.0150481135348186, 'r': 1.0012027604276916}
FEAT_HI = {'m1': 4.995768570369693, 'm2': 4.998228995967311, 'r': 4.998005409028107}

def features(m1, m2, r):
    """Preprocessing T (raw -> model features)."""
    m1 = m1
    m2 = m2
    r = r
    return dict(m1=m1, m2=m2, r=r)

def predict(m1, m2, r):
    f = features(m1, m2, r); m1 = f['m1']; m2 = f['m2']; r = f['r']
    return m1*m2/r**2

def in_domain(m1, m2, r):
    raw = dict(m1=m1, m2=m2, r=r); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
