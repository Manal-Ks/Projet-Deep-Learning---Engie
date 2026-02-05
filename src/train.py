from __future__ import annotations

import argparse
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor

from src.data import load_dataset, time_split
from src.features import build_preprocessor
from src.evaluate import evaluate
from src.utils import ensure_dir, save_json


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = REPO_ROOT / "data" / "engie_full.parquet"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"


def build_model(model_name: str):
    """
    Returns: (prep_mode, estimator)
      prep_mode: "linear" or "tree"
    """

    if model_name == "ridge":
        return "linear", Ridge(alpha=10.0, random_state=0)

    # True RandomForest baseline (light to avoid long training time)
    if model_name == "rf":
        return "tree", RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=50,
            max_features=0.5,
            n_jobs=-1,
            random_state=0,
        )

    # ExtraTrees = RF-like baseline, usually faster
    if model_name == "extratrees":
        return "tree", ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=0,
        )

    # Strong boosted trees baseline (sklearn-native replacement for XGBoost)
    if model_name == "hgb":
        return "tree", HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=800,
            random_state=0,
        )

    if model_name == "mlp":
        return "linear", MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            early_stopping=True,
            n_iter_no_change=20,
            max_iter=500,
            random_state=0,
        )

    raise ValueError("model must be one of: ridge, rf, extratrees, hgb, mlp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    ap.add_argument("--turbine", default="WT3", help="WT1/WT2/WT3/WT4 or None for all")
    ap.add_argument(
        "--model",
        required=True,
        choices=["ridge", "rf", "extratrees", "hgb", "mlp"],
        help="Model to train",
    )
    ap.add_argument("--out_dir", default=str(DEFAULT_MODELS_DIR))
    ap.add_argument(
        "--report_dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory where JSON results will be saved",
    )
    args = ap.parse_args()

    turbine = None if args.turbine.lower() == "none" else args.turbine

    df = load_dataset(args.parquet, turbine=turbine)
    X_train, y_train, X_val, y_val, X_test, y_test = time_split(df)

    prep_mode, estimator = build_model(args.model)
    preprocessor = build_preprocessor(X_train, mode=prep_mode)

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", estimator),
    ])

    pipe.fit(X_train, y_train)
    metrics = evaluate(pipe, X_val, y_val, X_test, y_test)

    out_dir = ensure_dir(args.out_dir)
    report_dir = ensure_dir(args.report_dir)

    model_path = out_dir / f"{args.model}_{args.turbine}.joblib"
    joblib.dump(pipe, model_path)

    report = {
        "model": args.model,
        "turbine": args.turbine,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        **metrics,
        "model_path": str(model_path),
        "parquet_path": str(Path(args.parquet)),
    }

    # Save one report per run to avoid overwriting
    report_path = report_dir / f"results_{args.model}_{args.turbine}.json"
    save_json(report, report_path)

    print(report)
    print(f"Saved model -> {model_path}")
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()
