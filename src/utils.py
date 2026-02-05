import pandas as pd
from sklearn.metrics import mean_absolute_error

def merge_xy(X, Y):
    return X.merge(Y, on="ID", how="inner")

def time_split(df, date_col="Date_time", train_frac=0.8):
    df = df.sort_values(date_col).reset_index(drop=True)
    cut = int(len(df) * train_frac)
    return df.iloc[:cut], df.iloc[cut:]

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

