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
    cap = pipe.params["total_credit_revolving_bal_cap"]
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
