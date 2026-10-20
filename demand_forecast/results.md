# Results (validation = 2017-01-01..2017-03-31, 500 store-item series, 45000 rows)

| model | SMAPE |
|---|---|
| seasonal naive (sales 364 days ago) | 19.154 |
| LightGBM (notebook recipe, 3000 trees) | 13.574 |
| **extracted formula** | **14.698** |

Formula vs LightGBM agreement (R² on log scale): 0.9892; formula size: 41 nodes (chosen: max_length 20, seed 0).

## Formula
log(1 + sales) = -0.1012*cos_month + 0.08941*dow_1 + 0.09584*dow_2 + 0.116*dow_3 + 0.1252*dow_4 + 0.1251*dow_5 + 0.127*dow_6 + 0.5554*log_ewm50_lag364 + 0.1062*log_lag_182 + 0.1737*log_lag_728 + 0.07613*log_lag_91 + 0.08426*log_lag_98 - 0.005302*sin_month - 0.02209

where log_X = log(1 + X) of each raw input, dow_d = 1 if day_of_week == d (Mon = 0 is base), sin_month/cos_month = sin/cos(2π·month/12).
sales = exp(log(1 + sales)) − 1.  Standalone code: `formula.py`.  Runtime: 322s
