"""Data: Kaggle 'Store Item Demand Forecasting Challenge' (10 stores x 50 items, daily sales 2013-2017).
Single fixed source (public GitHub copy of the Kaggle files)."""
import pandas as pd
SOURCE = "https://raw.githubusercontent.com/ugursaricam/store-item-demand-forecasting/master/datasets/"

def load_train(): return pd.read_csv(SOURCE + "train.csv", parse_dates=["date"])   # date, store, item, sales
def load_test():  return pd.read_csv(SOURCE + "test.csv", parse_dates=["date"])    # id, date, store, item (2018-01-01..03-31)
