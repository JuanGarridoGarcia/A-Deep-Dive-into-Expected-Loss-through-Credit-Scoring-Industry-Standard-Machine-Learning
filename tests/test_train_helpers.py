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
