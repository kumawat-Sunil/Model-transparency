"""Data: Kaggle 'Store Item Demand Forecasting Challenge' (10 stores x 50 items, daily sales 2013-2017), stored in data/."""
import pathlib, pandas as pd
DATA = pathlib.Path(__file__).resolve().parent / "data"

def load_train(): return pd.read_csv(DATA / "train.csv", parse_dates=["date"])   # date, store, item, sales
def load_test():  return pd.read_csv(DATA / "test.csv", parse_dates=["date"])    # id, date, store, item (2018-01-01..03-31)
