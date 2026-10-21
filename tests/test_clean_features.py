"""Tests for the PD and LGD outlier treatments."""

import numpy as np
import pandas as pd

from clean_features_LGD import clean_features_LGD
from clean_features_PD import clean_features_PD


def _pd_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "annual_income": rng.lognormal(11, 1.5, n),
        "used_credit_share": rng.uniform(0, 150, n),
        "bal_to_cred_lim": rng.uniform(0, 150, n),
        "num_inq_in_12mths": rng.integers(0, 40, n),
        "max_bal_owed": rng.lognormal(8, 1, n),
        "dept_paym_income_ratio": rng.normal(15, 10, n),
        "earliest_cr_line_year": rng.integers(1950, 2015, n),
    })
    df.loc[:9, "bal_to_cred_lim"] = np.nan
    return df


def test_clean_features_pd(capsys):
    df = _pd_frame()
    out = clean_features_PD(df)
    assert out["used_credit_share"].max() <= 100
    assert out["bal_to_cred_lim"].max() <= 100
    assert out["bal_to_cred_lim"].notna().all()
    assert (out["dept_paym_income_ratio"] >= 0).all()
    assert out["earliest_cr_line_year"].min() >= df["earliest_cr_line_year"].quantile(0.01)
    assert df["bal_to_cred_lim"].isna().sum() == 10  # input untouched


def test_clean_features_lgd(lgd_frame, capsys):
    out = clean_features_LGD(lgd_frame)
    assert out["used_credit_share"].max() <= 100
    cap = lgd_frame["num_open_credit_lines"].quantile(0.99)
    assert out["num_open_credit_lines"].max() <= cap
