"""tabular_regression / auto_mpg — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'cylinders': 3.0, 'displacement': 70.0, 'horsepower': 46.0, 'weight': 1613.0, 'acceleration': 8.5, 'model_year': 70.0, 'origin': 1.0}
RAW_HI = {'cylinders': 8.0, 'displacement': 455.0, 'horsepower': 225.0, 'weight': 5140.0, 'acceleration': 24.8, 'model_year': 82.0, 'origin': 3.0}
FEAT_LO = {'log_weight': 7.385851078125209, 'log_hp': 3.828641396489095, 'displacement': 70.0, 'acceleration': 8.5, 'model_year': 70.0, 'cylinders': 3.0, 'origin_2': 0.0, 'origin_3': -0.0}
FEAT_HI = {'log_weight': 8.544808358449211, 'log_hp': 5.41610040220442, 'displacement': 455.0, 'acceleration': 24.8, 'model_year': 82.0, 'cylinders': 8.0, 'origin_2': 1.0, 'origin_3': 1.0}

def features(cylinders, displacement, horsepower, weight, acceleration, model_year, origin):
    """Preprocessing T (raw -> model features)."""
    log_weight = np.log(weight)
    log_hp = np.log(horsepower)
    displacement = displacement
    acceleration = acceleration
    model_year = model_year
    cylinders = cylinders
    origin_2 = -(origin - 3)*(origin - 1)
    origin_3 = ((1/2)*origin - 1/2)*(origin - 2)
    return dict(log_weight=log_weight, log_hp=log_hp, displacement=displacement, acceleration=acceleration, model_year=model_year, cylinders=cylinders, origin_2=origin_2, origin_3=origin_3)

def predict(cylinders, displacement, horsepower, weight, acceleration, model_year, origin):
    f = features(cylinders, displacement, horsepower, weight, acceleration, model_year, origin); log_weight = f['log_weight']; log_hp = f['log_hp']; displacement = f['displacement']; acceleration = f['acceleration']; model_year = f['model_year']; cylinders = f['cylinders']; origin_2 = f['origin_2']; origin_3 = f['origin_3']
    return -7.0019e-5*acceleration*displacement - 0.041199*acceleration*log_hp + 0.057206*acceleration*origin_2 + 0.050574*acceleration*origin_3 + 0.0022529*cylinders**2 + 0.00075239*cylinders*displacement + 0.0073613*displacement*log_hp - 0.00044229*displacement*model_year + 0.0016144*displacement*origin_3 + 0.33234*log_hp*log_weight - 10.379*log_hp - 17.49*log_weight + 0.0054963*model_year**2 + 0.0047008*model_year*origin_3 + 0.17142*origin_2**2 + 0.17142*origin_2 + 167.319

def in_domain(cylinders, displacement, horsepower, weight, acceleration, model_year, origin):
    raw = dict(cylinders=cylinders, displacement=displacement, horsepower=horsepower, weight=weight, acceleration=acceleration, model_year=model_year, origin=origin); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
