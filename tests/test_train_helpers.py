"""Tests for the training helpers (skipped when the boosting libraries are missing)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("lightgbm")
pytest.importorskip("catboost")

from train_models_PD import _stratified_sample  # noqa: E402


def test_stratified_sample_keeps_class_balance():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=10_000)})
    y = pd.Series((rng.uniform(size=10_000) < 0.1).astype(int))
    Xs, ys = _stratified_sample(X, y, n=2_000)
    assert len(Xs) == len(ys)
    assert abs(len(Xs) - 2_000) <= 2
    assert ys.mean() == pytest.approx(y.mean(), abs=0.005)


def test_unknown_n_iter_override_is_rejected_before_training():
    from train_models_PD import train_all_models_PD

    X = pd.DataFrame({"a": [0.0, 1.0]})
    y = pd.Series([0, 1])
    with pytest.raises(ValueError, match="random_forrest"):
        train_all_models_PD(X, y, X, y, sample_size=2, n_iter_overrides={"random_forrest": 20})
