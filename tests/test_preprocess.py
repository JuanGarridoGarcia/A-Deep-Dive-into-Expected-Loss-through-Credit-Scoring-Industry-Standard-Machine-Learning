"""Tests for the shared scaling and encoding step."""

import numpy as np
import pandas as pd

from preprocess import preprocess_features, preprocess_features_LGD


def _frames():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0], "c": ["a", "b", "a", "b", "c"]})
    test = pd.DataFrame({"x": [2.5, 3.5], "c": ["a", "d"]})  # 'd' unseen, 'c' absent
    return train, test


def test_test_set_aligned_to_training_schema():
    train, test = _frames()
    tr, te, _ = preprocess_features(train, test, ["c"], ["x"])
    assert list(te.columns) == list(tr.columns)
    # Unseen category 'd' maps to all-zero dummies; absent 'c' is filled with 0.
    assert te.filter(like="c_").to_numpy().sum() == 0


def test_scaler_fitted_on_train_only():
    train, test = _frames()
    tr, te, scaler = preprocess_features(train, test, ["c"], ["x"])
    assert scaler.center_[0] == np.median(train["x"])
    np.testing.assert_allclose(scaler.transform(test[["x"]]).ravel(), te["x"].to_numpy())


def test_inputs_not_mutated():
    train, test = _frames()
    before = train.copy()
    preprocess_features(train, test, ["c"], ["x"])
    pd.testing.assert_frame_equal(train, before)


def test_lgd_alias_is_the_same_function():
    assert preprocess_features_LGD is preprocess_features


def test_apply_preprocessing_matches_the_fitted_transform():
    from preprocess import apply_preprocessing

    train, test = _frames()
    tr, te, scaler = preprocess_features(train, test, ["c"], ["x"])
    again = apply_preprocessing(test, ["c"], ["x"], tr.columns, scaler)
    pd.testing.assert_frame_equal(again, te)
