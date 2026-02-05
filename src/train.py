import argparse, os, json
import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor


def make_preprocess(X, linear: bool):
    cat_cols = ["MAC_CODE"] if "MAC_CODE" in X.columns else []
    num_cols = [c for c in X.columns if c not in cat_cols]

    if linear:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler())])
    else:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )


def load_split(parquet_path, turbine=None):
    df = pd.read_parquet(parquet_path)
    if turbine is not None:
        df = df[df["MAC_CODE"] == turbine].copy()

    df = df.sort_values("Date_time").reset_index(drop=True)

    target_col = "TARGET"
    drop_cols = ["ID", "Date_time", target_col]

    n = len(df)
    train = df.iloc[:int(0.70*n)]
    val   = df.iloc[int(0.70*n):int(0.85*n)]
    test  = df.iloc[int(0.85*n):]

    X_train = train.drop(columns=drop_cols)
    y_train = train[target_col]
    X_val = val.drop(columns=drop_cols)
    y_val = val[target_col]
    X_test = test.drop(columns=drop_cols)
    y_test = test[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="../data/engie_full.parquet")
    ap.add_argument("--model", choices=["ridge", "rf", "xgb", "mlp"], required=True)
    ap.add_argument("--turbine", default="WT3")  # set None for all turbines
    ap.add_argument("--out_dir", default="../models")
    args = ap.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_split(args.parquet, args.turbine)

    if args.model == "ridge":
        prep = make_preprocess(X_train, linear=True)
        model = Ridge(alpha=10.0, random_state=0)
    elif args.model == "rf":
        prep = make_preprocess(X_train, linear=False)
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1)
    elif args.model == "xgb":
        prep = make_preprocess(X_train, linear=False)
        model = XGBRegressor(
            n_estimators=2000, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
            random_state=0, n_jobs=-1
        )
    else:
        prep = make_preprocess(X_train, linear=True)
        model = MLPRegressor(
            hidden_layer_sizes=(256,128,64),
            early_stopping=True, n_iter_no_change=20, max_iter=500,
            learning_rate_init=1e-3, alpha=1e-4, random_state=0
        )

    pipe = Pipeline([("prep", prep), ("model", model)])

    pipe.fit(X_train, y_train)
    pred_val = pipe.predict(X_val)
    pred_test = pipe.predict(X_test)

    mae_val = float(mean_absolute_error(y_val, pred_val))
    mae_test = float(mean_absolute_error(y_test, pred_test))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(args.out_dir, f"{args.model}_{args.turbine}.joblib")
    joblib.dump(pipe, model_path)

    report = {"model": args.model, "turbine": args.turbine, "mae_val": mae_val, "mae_test": mae_test, "model_path": model_path}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

