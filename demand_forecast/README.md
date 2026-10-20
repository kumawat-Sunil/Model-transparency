# Demand forecast → formula

Data: Kaggle "Store Item Demand Forecasting Challenge" (10 stores × 50 items, daily 2013–2017), downloaded automatically from a public mirror into `data/`.

```
pip install -r requirements.txt
python train_and_extract.py   # LightGBM (Kaggle notebook recipe) -> symbolic regression -> formula.py + results.md  (~6 min)
python forecast.py            # forecasts 2018-01-01..03-31 using ONLY formula.py -> forecast_2018Q1.csv
```

| file | role |
|---|---|
| features.py | feature recipe (lags ≥ 91 days, lagged EWM/rolling means, calendar) + preprocessing used inside the formula |
| train_and_extract.py | trains the model, runs symbolic regression on its predictions, writes the formula |
| formula.py | generated standalone formula (numpy only) |
| forecast.py | applies the formula to new data |
| results.md | generated scores + formula |
