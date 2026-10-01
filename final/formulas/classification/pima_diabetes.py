"""classification / pima_diabetes — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'Pregnancies': 0.0, 'Glucose': 0.0, 'BloodPressure': 0.0, 'SkinThickness': 0.0, 'Insulin': 0.0, 'BMI': 0.0, 'DiabetesPedigreeFunction': 0.078, 'Age': 21.0}
RAW_HI = {'Pregnancies': 17.0, 'Glucose': 199.0, 'BloodPressure': 122.0, 'SkinThickness': 99.0, 'Insulin': 846.0, 'BMI': 67.1, 'DiabetesPedigreeFunction': 2.42, 'Age': 72.0}
FEAT_LO = {'Pregnancies': 0.0, 'Glucose': 0.0, 'BloodPressure': 0.0, 'SkinThickness': 0.0, 'Insulin': 0.0, 'BMI': 0.0, 'DiabetesPedigreeFunction': 0.078, 'Age': 21.0}
FEAT_HI = {'Pregnancies': 17.0, 'Glucose': 199.0, 'BloodPressure': 122.0, 'SkinThickness': 99.0, 'Insulin': 846.0, 'BMI': 67.1, 'DiabetesPedigreeFunction': 2.42, 'Age': 72.0}

def features(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    """Preprocessing T (raw -> model features)."""
    Pregnancies = Pregnancies
    Glucose = Glucose
    BloodPressure = BloodPressure
    SkinThickness = SkinThickness
    Insulin = Insulin
    BMI = BMI
    DiabetesPedigreeFunction = DiabetesPedigreeFunction
    Age = Age
    return dict(Pregnancies=Pregnancies, Glucose=Glucose, BloodPressure=BloodPressure, SkinThickness=SkinThickness, Insulin=Insulin, BMI=BMI, DiabetesPedigreeFunction=DiabetesPedigreeFunction, Age=Age)

def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    f = features(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age); Pregnancies = f['Pregnancies']; Glucose = f['Glucose']; BloodPressure = f['BloodPressure']; SkinThickness = f['SkinThickness']; Insulin = f['Insulin']; BMI = f['BMI']; DiabetesPedigreeFunction = f['DiabetesPedigreeFunction']; Age = f['Age']
    return 1 / (1 + np.exp(-(-0.0029859*Age**2 - 0.0027156*Age*BMI + 0.0020647*Age*BloodPressure - 0.024134*Age*DiabetesPedigreeFunction + 3.9418e-5*Age*Glucose + 0.00023553*Age*Insulin - 0.018084*Age*Pregnancies - 0.00052169*Age*SkinThickness + 0.25885*Age - 0.0015246*BMI**2 - 0.043153*BMI*DiabetesPedigreeFunction - 0.00064252*BMI*Glucose - 9.4694e-5*BMI*Insulin - 0.012748*BMI*Pregnancies + 0.44874*BMI - 0.00018231*BloodPressure**2 - 0.035843*BloodPressure*DiabetesPedigreeFunction - 0.00033628*BloodPressure*Glucose + 0.00027532*BloodPressure*Insulin - 0.0019012*BloodPressure*Pregnancies + 0.0002689*BloodPressure*SkinThickness - 0.41574*DiabetesPedigreeFunction**2 - 0.019008*DiabetesPedigreeFunction*Glucose - 0.0032215*DiabetesPedigreeFunction*Insulin + 0.026287*DiabetesPedigreeFunction*Pregnancies + 0.045044*DiabetesPedigreeFunction*SkinThickness + 8.2027*DiabetesPedigreeFunction - 6.5014e-5*Glucose**2 + 3.3696e-5*Glucose*Insulin - 0.0037922*Glucose*Pregnancies - 0.00073615*Glucose*SkinThickness + 0.15615*Glucose + 1.155e-5*Insulin**2 + 0.00014438*Insulin*Pregnancies + 0.00015088*Insulin*SkinThickness - 0.03784*Insulin + 0.026784*Pregnancies**2 + 0.0016993*Pregnancies*SkinThickness + 1.5184*Pregnancies + 0.0002721*SkinThickness**2 + 0.055238*SkinThickness - 32.8489)))

def in_domain(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    raw = dict(Pregnancies=Pregnancies, Glucose=Glucose, BloodPressure=BloodPressure, SkinThickness=SkinThickness, Insulin=Insulin, BMI=BMI, DiabetesPedigreeFunction=DiabetesPedigreeFunction, Age=Age); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
