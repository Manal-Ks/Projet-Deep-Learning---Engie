from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X, mode: str):
    """
    Build a preprocessing pipeline.

    Parameters
    ----------
    X : pandas.DataFrame
        Training features (used to infer numeric/categorical columns).
    mode : str
        - "linear": impute + scale numeric features (for Ridge/MLP)
        - "tree"  : impute numeric features only (for tree models)

    Returns
    -------
    sklearn ColumnTransformer
    """
    cat_cols = ["MAC_CODE"] if "MAC_CODE" in X.columns else []
    num_cols = [c for c in X.columns if c not in cat_cols]

    if mode == "linear":
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    elif mode == "tree":
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
    else:
        raise ValueError("mode must be 'linear' or 'tree'")

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


