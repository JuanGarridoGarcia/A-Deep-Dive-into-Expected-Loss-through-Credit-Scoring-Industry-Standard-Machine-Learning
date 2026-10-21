"""Tests for FeaturePipeline_LGD (fit on defaulted loans, apply to any population)."""

import numpy as np
import pandas as pd
import pytest

from conftest import LGD_NUMERIC
from pipeline_LGD import FeaturePipeline_LGD


def test_transform_before_fit_raises(lgd_frame):
    with pytest.raises(RuntimeError, match="not been fitted"):
        FeaturePipeline_LGD().transform(lgd_frame)


def test_parameters_come_from_fit_population_only(lgd_frame):
    pipe = FeaturePipeline_LGD().fit(lgd_frame)
    cap = pipe.params["treatment"]["total_credit_revolving_bal"]["cap_upper"]
    assert cap == pytest.approx(lgd_frame["total_credit_revolving_bal"].quantile(0.99))

    # A new population with much larger balances must be capped at the fitted value,
    # not at its own 99th percentile.
    other = lgd_frame.copy()
    other["total_credit_revolving_bal"] *= 10
    out_unscaled = pipe._apply_treatments(other.copy())
    assert out_unscaled["total_credit_revolving_bal"].max() == pytest.approx(cap)


def test_output_schema_is_stable(lgd_frame):
    pipe = FeaturePipeline_LGD().fit(lgd_frame)
    only_36 = lgd_frame[lgd_frame["loan_term_months"] == 36]
    out = pipe.transform(only_36)
    assert list(out.columns) == pipe.params["encoded_columns"]
    assert out["loan_term_months_60"].eq(0).all()


def test_economic_ceiling_and_scaling(lgd_frame):
    pipe = FeaturePipeline_LGD().fit(lgd_frame)
    treated = pipe._apply_treatments(lgd_frame.copy())
    assert treated["used_credit_share"].max() <= 100
    out = pipe.transform(lgd_frame)
    # RobustScaler: fitted medians map to zero.
    np.testing.assert_allclose(out[LGD_NUMERIC].median(), 0, atol=1e-9)


def test_save_and_load_roundtrip(lgd_frame, tmp_path):
    pipe = FeaturePipeline_LGD().fit(lgd_frame)
    path = tmp_path / "pipeline.pkl"
    pipe.save(path)
    loaded = FeaturePipeline_LGD.load(path)
    pd.testing.assert_frame_equal(loaded.transform(lgd_frame), pipe.transform(lgd_frame))


def test_pipeline_follows_the_selected_feature_set(lgd_frame):
    # A different selection: a feature without a treatment step and a text category.
    df = lgd_frame.drop(columns=["num_open_credit_lines"]).assign(
        tot_num_credit_lines=np.arange(len(lgd_frame)) % 50,
        verification_status=np.where(np.arange(len(lgd_frame)) % 3 == 0, "Verified", "Not Verified"),
    )
    pipe = FeaturePipeline_LGD().fit(df)
    out = pipe.transform(df)
    assert "verification_status_Verified" in out.columns
    assert "tot_num_credit_lines" in pipe.params["numeric_features"]
    assert "num_open_credit_lines" not in out.columns
    assert out.isna().sum().sum() == 0


def test_missing_values_are_imputed_with_the_fit_median(lgd_frame):
    df = lgd_frame.copy()
    df.loc[:19, "dept_paym_income_ratio"] = np.nan
    pipe = FeaturePipeline_LGD().fit(df)
    treated = pipe._apply_treatments(df)
    expected = df["dept_paym_income_ratio"].clip(lower=0).median()
    assert treated.loc[:19, "dept_paym_income_ratio"].eq(expected).all()
