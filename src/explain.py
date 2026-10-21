"""
explain.py

SHAP values for whichever model the comparison selects.

The PD and LGD notebooks pick the model with the best cross-validated score, and
that can be a linear model (logistic regression, ridge) as well as a tree
ensemble. TreeExplainer only accepts tree models, so the explainer is chosen
from the fitted model instead of being fixed in the notebook.
"""

import numpy as np
import pandas as pd
import shap


def shap_values_for(model, X_explain: pd.DataFrame, X_background: pd.DataFrame,
                    background_size: int = 10_000, random_state: int = 42) -> np.ndarray:
    """
    SHAP values of `model` on `X_explain`, as a (rows x features) array.

    Linear models (anything with a `coef_` attribute) use LinearExplainer with an
    independent masker built from a sample of `X_background` (the training split).
    For a logistic regression the values are in log-odds, the scale on which the
    model is additive. Tree models use TreeExplainer, which is exact and needs no
    background data. For binary classifiers the positive class (default) is returned.
    """
    X_explain = X_explain.astype(float)

    if hasattr(model, "coef_"):
        n = min(background_size, len(X_background))
        background = X_background.sample(n=n, random_state=random_state).astype(float)
        explainer = shap.LinearExplainer(model, shap.maskers.Independent(background, max_samples=n))
    else:
        explainer = shap.TreeExplainer(model)

    values = explainer.shap_values(X_explain)
    if isinstance(values, list):          # older SHAP: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                  # newer SHAP: (rows, features, classes)
        values = values[..., 1]
    return values
