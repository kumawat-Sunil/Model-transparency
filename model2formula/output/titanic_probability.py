"""titanic_probability — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'pclass': 1.0, 'sex': 0.0, 'age': 0.42, 'sibsp': 0.0, 'fare': 0.0}
RAW_HI = {'pclass': 3.0, 'sex': 1.0, 'age': 80.0, 'sibsp': 8.0, 'fare': 512.3292}
FEAT_LO = {'pclass': 1.0, 'sex': 0.0, 'age': 0.42, 'sibsp': 0.0, 'log_fare': 0.0}
FEAT_HI = {'pclass': 3.0, 'sex': 1.0, 'age': 80.0, 'sibsp': 8.0, 'log_fare': 6.240917354759096}

def features(pclass, sex, age, sibsp, fare):
    """Preprocessing T (raw -> model features)."""
    pclass = pclass
    sex = sex
    age = age
    sibsp = sibsp
    log_fare = np.log(fare + 1)
    return dict(pclass=pclass, sex=sex, age=age, sibsp=sibsp, log_fare=log_fare)

def predict(pclass, sex, age, sibsp, fare):
    f = features(pclass, sex, age, sibsp, fare); pclass = f['pclass']; sex = f['sex']; age = f['age']; sibsp = f['sibsp']; log_fare = f['log_fare']
    return 1 / (1 + np.exp(-(0.00084546*age**2 + 0.010766*age*log_fare - 0.0038473*age*pclass + 0.0047587*age*sibsp - 0.12072*age - 0.049737*log_fare**2 - 0.028729*log_fare*pclass - 0.91951*pclass*sex - 0.10905*pclass*sibsp - 0.51676*pclass + 2.4415*sex**2 - 0.15375*sex*sibsp + 2.4415*sex - 0.046537*sibsp**2 + 2.47615)))

def in_domain(pclass, sex, age, sibsp, fare):
    raw = dict(pclass=pclass, sex=sex, age=age, sibsp=sibsp, fare=fare); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
