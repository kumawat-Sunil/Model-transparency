"""diabetes — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'age': 19.0, 'sex': 1.0, 'bmi': 18.0, 'bp': 62.0, 's1': 97.0, 's2': 41.6, 's3': 25.0, 's4': 2.0, 's5': 3.4965, 's6': 58.0}
RAW_HI = {'age': 79.0, 'sex': 2.0, 'bmi': 42.2, 'bp': 131.0, 's1': 300.0, 's2': 242.4, 's3': 99.0, 's4': 9.09, 's5': 6.107, 's6': 124.0}
FEAT_LO = {'age': 19.0, 'sex': 1.0, 'bmi': 18.0, 'bp': 62.0, 's1': 97.0, 's2': 41.6, 's3': 25.0, 's4': 2.0, 's5': 3.4965, 's6': 58.0}
FEAT_HI = {'age': 79.0, 'sex': 2.0, 'bmi': 42.2, 'bp': 131.0, 's1': 300.0, 's2': 242.4, 's3': 99.0, 's4': 9.09, 's5': 6.107, 's6': 124.0}

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
    return 0.0057658*age**2 + 0.0058568*age*bmi - 0.0031565*age*s1 - 0.0063045*age*s3 + 0.046579*age*s4 + 0.21177*age*sex + 0.026191*bmi*bp + 0.57421*bmi*s5 + 0.029822*bp*s4 + 0.053356*bp*s5 - 3.0383e-5*s1**2 + 0.0006549*s1*s3 - 0.00030223*s1*s6 - 0.018702*s1*sex - 0.056056*s1 - 0.0016889*s2*s6 - 0.0062928*s2*sex - 0.0012106*s3*s6 + 0.066104*s3*sex - 0.2463*s3 + 0.13973*s4**2 + 41.624*s5 - 5.5161*sex**2 - 16.548*sex - 133.252

def in_domain(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6):
    raw = dict(age=age, sex=sex, bmi=bmi, bp=bp, s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, s6=s6); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
