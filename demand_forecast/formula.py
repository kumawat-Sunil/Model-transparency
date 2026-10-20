"""Demand formula extracted from the LightGBM model by symbolic regression (numpy only).
predict(lag_91, lag_98, lag_182, lag_364, lag_728, ewm95_lag91, ewm50_lag91, ewm50_lag364, roll365_lag91, dow, month) -> daily sales"""
import numpy as np

def predict(lag_91, lag_98, lag_182, lag_364, lag_728, ewm95_lag91, ewm50_lag91, ewm50_lag364, roll365_lag91, dow, month):
    log_lag_91 = np.log(lag_91 + 1)
    log_lag_98 = np.log(lag_98 + 1)
    log_lag_182 = np.log(lag_182 + 1)
    log_lag_364 = np.log(lag_364 + 1)
    log_lag_728 = np.log(lag_728 + 1)
    log_ewm95_lag91 = np.log(ewm95_lag91 + 1)
    log_ewm50_lag91 = np.log(ewm50_lag91 + 1)
    log_ewm50_lag364 = np.log(ewm50_lag364 + 1)
    log_roll365_lag91 = np.log(roll365_lag91 + 1)
    dow_1 = -1/120*dow*(dow - 6)*(dow - 5)*(dow - 4)*(dow - 3)*(dow - 2)
    dow_2 = (1/48)*dow*(dow - 6)*(dow - 5)*(dow - 4)*(dow - 3)*(dow - 1)
    dow_3 = -1/36*dow*(dow - 6)*(dow - 5)*(dow - 4)*(dow - 2)*(dow - 1)
    dow_4 = (1/48)*dow*(dow - 6)*(dow - 5)*(dow - 3)*(dow - 2)*(dow - 1)
    dow_5 = -1/120*dow*(dow - 6)*(dow - 4)*(dow - 3)*(dow - 2)*(dow - 1)
    dow_6 = (1/720)*dow*(dow - 5)*(dow - 4)*(dow - 3)*(dow - 2)*(dow - 1)
    sin_month = np.sin((1/6)*np.pi*month)
    cos_month = np.cos((1/6)*np.pi*month)
    log_sales = -0.10123*cos_month + 0.089407*dow_1 + 0.095837*dow_2 + 0.11605*dow_3 + 0.12522*dow_4 + 0.12515*dow_5 + 0.12698*dow_6 + 0.55542*log_ewm50_lag364 + 0.10619*log_lag_182 + 0.17367*log_lag_728 + 0.076126*log_lag_91 + 0.08426*log_lag_98 - 0.0053017*sin_month - 0.022092
    return np.expm1(log_sales)
