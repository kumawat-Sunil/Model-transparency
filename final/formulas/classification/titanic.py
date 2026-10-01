"""classification / titanic — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'pclass': 1.0, 'sex': 0.0, 'age': 0.42, 'sibsp': 0.0, 'fare': 0.0}
RAW_HI = {'pclass': 3.0, 'sex': 1.0, 'age': 71.0, 'sibsp': 8.0, 'fare': 512.3292}
FEAT_LO = {'pclass': 1.0, 'sex': 0.0, 'age': 0.42, 'sibsp': 0.0, 'log_fare': 0.0}
FEAT_HI = {'pclass': 3.0, 'sex': 1.0, 'age': 71.0, 'sibsp': 8.0, 'log_fare': 6.240917354759096}

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
    return 1 / (1 + np.exp(-(-0.017073*age*pclass + 0.024474*age*sex + 0.0013461*age*sibsp - 0.021699*age - 0.098385*log_fare**2 - 0.26043*log_fare*sex - 0.11789*log_fare*sibsp + 1.0303*log_fare - 1.4084*pclass*sex - 0.24694*pclass*sibsp + 2.9596*sex**2 + 0.083082*sex*sibsp + 2.9596*sex - 0.066853*sibsp**2 + 0.79612*sibsp - 1.62724)))

def in_domain(pclass, sex, age, sibsp, fare):
    raw = dict(pclass=pclass, sex=sex, age=age, sibsp=sibsp, fare=fare); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
