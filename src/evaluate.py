from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def mape(y_true, y_pred, eps=1e-8):
    """Mean Absolute Percentage Error (safe version)"""
    y_true = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate(pipe, X_val, y_val, X_test, y_test) -> dict:
    """
    Compute multiple regression metrics on validation and test sets.
    """

    pred_val = pipe.predict(X_val)
    pred_test = pipe.predict(X_test)

    results = {
        # Validation
        "mae_val": float(mean_absolute_error(y_val, pred_val)),
        "rmse_val": float(np.sqrt(mean_squared_error(y_val, pred_val))),
        "r2_val": float(r2_score(y_val, pred_val)),
        "mape_val": float(mape(y_val, pred_val)),

        # Test
        "mae_test": float(mean_absolute_error(y_test, pred_test)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "r2_test": float(r2_score(y_test, pred_test)),
        "mape_test": float(mape(y_test, pred_test)),
    }

    return results

