"""tabular_regression / boston — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'crim': 0.00632, 'rm': 3.561, 'age': 2.9, 'dis': 1.1296, 'tax': 188.0, 'ptratio': 12.6, 'lstat': 1.73, 'nox': 0.385}
RAW_HI = {'crim': 88.9762, 'rm': 8.78, 'age': 100.0, 'dis': 10.7103, 'tax': 711.0, 'ptratio': 22.0, 'lstat': 37.97, 'nox': 0.871}
FEAT_LO = {'log_crim': 0.006300112548479956, 'rm': 3.561, 'age': 2.9, 'log_dis': 0.12186358775684052, 'tax': 188.0, 'ptratio': 12.6, 'log_lstat': 0.5481214085096876, 'nox': 0.385}
FEAT_HI = {'log_crim': 4.4995451909142234, 'rm': 8.78, 'age': 100.0, 'log_dis': 2.371205895271833, 'tax': 711.0, 'ptratio': 22.0, 'log_lstat': 3.636796374243711, 'nox': 0.871}

def features(crim, rm, age, dis, tax, ptratio, lstat, nox):
    """Preprocessing T (raw -> model features)."""
    log_crim = np.log(crim + 1)
    rm = rm
    age = age
    log_dis = np.log(dis)
    tax = tax
    ptratio = ptratio
    log_lstat = np.log(lstat)
    nox = nox
    return dict(log_crim=log_crim, rm=rm, age=age, log_dis=log_dis, tax=tax, ptratio=ptratio, log_lstat=log_lstat, nox=nox)

def predict(crim, rm, age, dis, tax, ptratio, lstat, nox):
    f = features(crim, rm, age, dis, tax, ptratio, lstat, nox); log_crim = f['log_crim']; rm = f['rm']; age = f['age']; log_dis = f['log_dis']; tax = f['tax']; ptratio = f['ptratio']; log_lstat = f['log_lstat']; nox = f['nox']
    return -1.4873e-5*age**2 + 0.001152*age*log_crim - 0.0036427*age*log_dis - 0.014443*age*log_lstat + 0.016666*age*nox + 0.00038282*age*ptratio - 0.0039031*age*rm - 1.9356e-5*age*tax - 0.60122*log_crim**2 + 0.13612*log_crim*log_dis - 1.4531*log_crim*log_lstat - 3.647*log_crim*nox + 0.028981*log_crim*ptratio - 0.82852*log_crim*rm + 12.022*log_crim + 1.2539*log_dis*log_lstat - 0.64129*log_dis*nox - 0.08593*log_dis*ptratio - 0.3524*log_dis*rm - 0.0028934*log_dis*tax - 3.0751*log_dis + 0.27589*log_lstat**2 - 3.3851*log_lstat*nox - 1.5717*log_lstat*rm - 0.005841*log_lstat*tax + 7.2871*log_lstat - 0.82984*nox*rm + 0.0083361*nox*tax - 0.0032995*ptratio**2 + 0.00028094*ptratio*tax - 0.61335*ptratio + 0.19867*rm**2 + 7.0812*rm + 7.1964e-6*tax**2 + 4.87444

def in_domain(crim, rm, age, dis, tax, ptratio, lstat, nox):
    raw = dict(crim=crim, rm=rm, age=age, dis=dis, tax=tax, ptratio=ptratio, lstat=lstat, nox=nox); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
