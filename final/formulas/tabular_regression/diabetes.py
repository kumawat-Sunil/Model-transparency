"""tabular_regression / diabetes — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'age': 19.0, 'sex': 1.0, 'bmi': 18.0, 'bp': 65.0, 's1': 110.0, 's2': 41.6, 's3': 23.0, 's4': 2.0, 's5': 3.2581, 's6': 58.0}
RAW_HI = {'age': 79.0, 'sex': 2.0, 'bmi': 42.2, 'bp': 133.0, 's1': 301.0, 's2': 242.4, 's3': 99.0, 's4': 9.09, 's5': 6.107, 's6': 124.0}
FEAT_LO = {'age': 19.0, 'sex': 1.0, 'bmi': 18.0, 'bp': 65.0, 's1': 110.0, 's2': 41.6, 's3': 23.0, 's4': 2.0, 's5': 3.2581, 's6': 58.0}
FEAT_HI = {'age': 79.0, 'sex': 2.0, 'bmi': 42.2, 'bp': 133.0, 's1': 301.0, 's2': 242.4, 's3': 99.0, 's4': 9.09, 's5': 6.107, 's6': 124.0}

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
    return 0.018493*age*bmi - 0.0038813*age*s1 + 0.016144*age*s4 + 0.0043289*age*s6 + 0.023635*bmi*bp - 0.010151*bmi*s4 + 0.42052*bmi*s5 + 0.0047346*bmi*s6 + 0.041636*bp*s4 + 0.082893*bp*s5 - 0.0018554*s1*s3 + 0.0012955*s1*sex + 0.034139*s1 - 0.01028*s2*sex + 36.33*s5 - 2.8795*sex**2 - 8.6384*sex - 166.772

def in_domain(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6):
    raw = dict(age=age, sex=sex, bmi=bmi, bp=bp, s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, s6=s6); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
