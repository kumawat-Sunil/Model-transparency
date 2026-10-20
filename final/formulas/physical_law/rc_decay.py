"""physical_law / rc_decay — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'V0': 1.0003827292937455, 't': 1.0082273332856402, 'R': 1.0102077257996216}
RAW_HI = {'V0': 4.999337044829548, 't': 4.984367790854354, 'R': 4.999372612561807}
FEAT_LO = {'V0': 1.0003827292937455, 't': 1.0082273332856402, 'R': 1.0102077257996216}
FEAT_HI = {'V0': 4.999337044829548, 't': 4.984367790854354, 'R': 4.999372612561807}

def features(V0, t, R):
    """Preprocessing T (raw -> model features)."""
    V0 = V0
    t = t
    R = R
    return dict(V0=V0, t=t, R=R)

def predict(V0, t, R):
    f = features(V0, t, R); V0 = f['V0']; t = f['t']; R = f['R']
    return -0.3417*V0*np.log(np.sin(0.352*t/R))

def in_domain(V0, t, R):
    raw = dict(V0=V0, t=t, R=R); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
