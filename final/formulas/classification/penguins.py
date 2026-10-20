"""classification / penguins — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'bill_length_mm': 33.5, 'bill_depth_mm': 13.1, 'flipper_length_mm': 172.0, 'body_mass_g': 2700.0}
RAW_HI = {'bill_length_mm': 58.0, 'bill_depth_mm': 21.5, 'flipper_length_mm': 231.0, 'body_mass_g': 6000.0}
FEAT_LO = {'bill_length_mm': 33.5, 'bill_depth_mm': 13.1, 'flipper_length_mm': 172.0, 'body_mass_g': 2700.0}
FEAT_HI = {'bill_length_mm': 58.0, 'bill_depth_mm': 21.5, 'flipper_length_mm': 231.0, 'body_mass_g': 6000.0}

def features(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    """Preprocessing T (raw -> model features)."""
    bill_length_mm = bill_length_mm
    bill_depth_mm = bill_depth_mm
    flipper_length_mm = flipper_length_mm
    body_mass_g = body_mass_g
    return dict(bill_length_mm=bill_length_mm, bill_depth_mm=bill_depth_mm, flipper_length_mm=flipper_length_mm, body_mass_g=body_mass_g)

def predict(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    f = features(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g); bill_length_mm = f['bill_length_mm']; bill_depth_mm = f['bill_depth_mm']; flipper_length_mm = f['flipper_length_mm']; body_mass_g = f['body_mass_g']
    return 1 / (1 + np.exp(-(-0.1504*bill_depth_mm**2 + 0.045678*bill_depth_mm*bill_length_mm + 0.00014401*bill_depth_mm*body_mass_g + 0.012808*bill_depth_mm*flipper_length_mm + 0.0091445*bill_length_mm**2 - 0.00021635*bill_length_mm*body_mass_g - 0.0078572*bill_length_mm*flipper_length_mm + 1.6041*bill_length_mm + 1.7108e-5*body_mass_g*flipper_length_mm - 39.9296)))

def in_domain(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    raw = dict(bill_length_mm=bill_length_mm, bill_depth_mm=bill_depth_mm, flipper_length_mm=flipper_length_mm, body_mass_g=body_mass_g); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
