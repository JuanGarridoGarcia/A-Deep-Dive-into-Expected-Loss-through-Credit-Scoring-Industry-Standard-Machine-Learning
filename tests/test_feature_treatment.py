"""Tests for the specification-driven feature treatment."""

import numpy as np
import pandas as pd
import pytest

from feature_treatment import FeatureTreatment

SPEC = {
    "income": [("cap_upper_quantile", 0.9), ("log1p_if_skewed", 2.0)],
    "share": [("cap_upper_value", 100)],
    "ratio": [("clip_lower_value", 0)],
    "year": [("cap_lower_quantile", 0.1)],
    "home": [("replace", {"OTHER": "RENT"})],
}


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "income": rng.lognormal(10, 1.5, n),
        "share": rng.uniform(0, 150, n),
        "ratio": rng.normal(5, 10, n),
        "year": rng.integers(1950, 2015, n),
        "home": rng.choice(["RENT", "OWN", "OTHER"], n),
        "untreated": rng.normal(size=n),
    })


def test_parameters_are_fitted_on_the_reference_only():
    ref, other = _frame(seed=0), _frame(seed=1)
    ft = FeatureTreatment(SPEC, verbose=False).fit(ref)
    out = ft.transform(other)
    cap = ref["income"].quantile(0.9)
    expected_max = np.log1p(cap) if ft.params["income"]["log1p"] else cap
    assert out["income"].max() <= expected_max + 1e-9
    assert out["year"].min() >= ref["year"].quantile(0.1)


def test_fixed_rules_and_recoding():
    out = FeatureTreatment(SPEC, verbose=False).fit_transform(_frame())
    assert out["share"].max() <= 100
    assert (out["ratio"] >= 0).all()
    assert "OTHER" not in set(out["home"])


def test_only_present_features_are_treated_and_others_kept():
    df = _frame()[["share", "untreated"]]
    out = FeatureTreatment(SPEC, verbose=False).fit_transform(df)
    assert list(out.columns) == ["share", "untreated"]
    pd.testing.assert_series_equal(out["untreated"], df["untreated"])


def test_numeric_nans_are_imputed_with_the_fit_median():
    df = _frame()
    df.loc[:9, "untreated"] = np.nan
    ft = FeatureTreatment(SPEC, verbose=False).fit(df)
    out = ft.transform(df)
    assert out["untreated"].notna().all()
    assert out.loc[0, "untreated"] == pytest.approx(df["untreated"].median())


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not been fitted"):
        FeatureTreatment(SPEC).transform(_frame())


def test_unknown_step_is_rejected():
    with pytest.raises(ValueError, match="Unknown treatment"):
        FeatureTreatment({"share": [("winsorize", 0.9)]}).fit(_frame())
