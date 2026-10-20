"""diabetes formula (trial t05) — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'age': 19.0, 'sex': 1.0, 'bmi': 18.0, 'bp': 62.0, 's1': 113.0, 's2': 43.4, 's3': 22.0, 's4': 2.0, 's5': 3.2581, 's6': 60.0}
RAW_HI = {'age': 79.0, 'sex': 2.0, 'bmi': 42.2, 'bp': 133.0, 's1': 301.0, 's2': 215.0, 's3': 93.0, 's4': 8.28, 's5': 6.107, 's6': 124.0}
FEAT_LO = {'age': 19.0, 'sex': 1.0, 'bmi': 18.0, 'bp': 62.0, 's1': 113.0, 's2': 43.4, 's3': 22.0, 's4': 2.0, 's5': 3.2581, 's6': 60.0}
FEAT_HI = {'age': 79.0, 'sex': 2.0, 'bmi': 42.2, 'bp': 133.0, 's1': 301.0, 's2': 215.0, 's3': 93.0, 's4': 8.28, 's5': 6.107, 's6': 124.0}

def features(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6):
    """Preprocessing T (raw -> model features)."""
    age = age
    sex = sex
    bmi = bmi
    bp = bp
    s1 = s1
    s2 = s2
    s3 = s3
    s4 = s4
    s5 = s5
    s6 = s6
    return dict(age=age, sex=sex, bmi=bmi, bp=bp, s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, s6=s6)

def predict(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6):
    f = features(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6); age = f['age']; sex = f['sex']; bmi = f['bmi']; bp = f['bp']; s1 = f['s1']; s2 = f['s2']; s3 = f['s3']; s4 = f['s4']; s5 = f['s5']; s6 = f['s6']
    return 1.226*bmi*s5 + 1.2*bp - 0.6334*s1 + 0.4728*s2 + 5.36*s4 + 1.73*s5**2 - 21.55*sex - 73

def in_domain(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6):
    raw = dict(age=age, sex=sex, bmi=bmi, bp=bp, s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, s6=s6); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
