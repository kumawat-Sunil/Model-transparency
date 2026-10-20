# Knee formulas (XGB teacher -> sparse symbolic -> fused with T; raw symbols l1,l2,l3,l6,l12,month)

## airline (3 terms)

In feature space: `15.03*lg1 + 10.97*lg12**2 + 51.58*lg12 + 215.2`

Fused on raw: `49.817*log(l1 + 1) + 126.42*log(l12 + 1)**2 - 1151.6*log(l12 + 1) + 2509.3`

## co2 (5 terms)

In feature space: `11.71*lg1 + 0.3013*lg12**2 + 3.062*lg6 - 0.3686*sin2_m + 1.264*sin_m + 337.1`

Fused on raw: `0.30133*(23.215*log(l12 + 1) - 135.11)**2 + 268.36*log(l1 + 1) + 70.39*log(l6 + 1) + 1.7953*sin(pi*month/6) - 0.52161*sin(pi*month/3) - 1635.3`

## sunspots (3 terms)

In feature space: `15.68*lg1 + 15.88*lgroll3**2 + 44.92*lgroll3 + 59.78`

Fused on raw: `15.676*log(l1 + 1) + 18.363*log(l1/3 + l2/3 + l3/3 + 1)**2 - 99.061*log(l1/3 + l2/3 + l3/3 + 1) + 99.165`

## carsales (4 terms)

In feature space: `-745.6*cos_m*lg12 + 572.9*lg1 + 2711.0*lg12 - 576.5*sin2_m + 1.282e+4`

Fused on raw: `(22750.0 - 2436.7*log(l12 + 1))*(1.4142*cos(pi*month/6) + 5.2336e-17) + 1835.1*log(l1 + 1) + 8859.8*log(l12 + 1) - 815.25*sin(pi*month/3) - 87203.0`
