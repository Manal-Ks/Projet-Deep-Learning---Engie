from __future__ import annotations
import pandas as pd

TARGET_COL = "TARGET"
DROP_COLS = ["ID", "Date_time", TARGET_COL]

def load_dataset(parquet_path: str, turbine: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if turbine is not None:
        df = df[df["MAC_CODE"] == turbine].copy()
    df = df.sort_values("Date_time").reset_index(drop=True)
    return df

def time_split(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    n = len(df)
    i1 = int(train_ratio * n)
    i2 = int((train_ratio + val_ratio) * n)

    train = df.iloc[:i1]
    val = df.iloc[i1:i2]
    test = df.iloc[i2:]

    X_train = train.drop(columns=DROP_COLS)
    y_train = train[TARGET_COL]
    X_val = val.drop(columns=DROP_COLS)
    y_val = val[TARGET_COL]
    X_test = test.drop(columns=DROP_COLS)
    y_test = test[TARGET_COL]

    return X_train, y_train, X_val, y_val, X_test, y_test

