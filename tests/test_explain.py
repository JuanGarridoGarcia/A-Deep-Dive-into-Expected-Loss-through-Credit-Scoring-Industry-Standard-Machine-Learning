"""Tests for explain.shap_values_for: one SHAP path per model family."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

pytest.importorskip("shap")
from explain import shap_values_for  # noqa: E402


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "x1": rng.normal(size=500),
        "x2": rng.normal(size=500),
        "flag": rng.integers(0, 2, size=500).astype(bool),   # one-hot columns are bool
    })
    logit = 1.5 * X["x1"] - 0.5 * X["x2"] + 0.8 * X["flag"]
    y_class = (rng.uniform(size=500) < 1 / (1 + np.exp(-logit))).astype(int)
    y_reg = 2.0 * X["x1"] + X["flag"] + rng.normal(scale=0.1, size=500)
    return X, y_class, y_reg


def test_logistic_regression_is_explained_in_log_odds(data):
    X, y, _ = data
    model = LogisticRegression().fit(X.astype(float), y)
    values = shap_values_for(model, X.iloc[:50], X)
    assert values.shape == (50, 3)
    # With an independent masker, a linear model's SHAP value is coef * (x - background mean)
    expected = model.coef_[0] * (X.iloc[:50].astype(float) - X.astype(float).mean()).to_numpy()
    np.testing.assert_allclose(values, expected, atol=1e-6)


def test_ridge_regression(data):
    X, _, y = data
    model = Ridge().fit(X.astype(float), y)
    assert shap_values_for(model, X.iloc[:50], X).shape == (50, 3)


@pytest.mark.parametrize("model_cls, target", [(DecisionTreeClassifier, 1), (DecisionTreeRegressor, 2)])
def test_tree_models(data, model_cls, target):
    X, y_class, y_reg = data
    y = y_class if target == 1 else y_reg
    model = model_cls(max_depth=3, random_state=0).fit(X, y)
    assert shap_values_for(model, X.iloc[:50], X).shape == (50, 3)
