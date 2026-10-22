"""Tests for lgd_analysis: the LGD decomposition and the variance shares."""

import numpy as np
import pandas as pd
import pytest

from lgd_analysis import lgd_by_months_paid, lgd_components, variance_explained


def _loans():
    # Two loans of 10,000 with the same 20% recovery rate, defaulting at different times
    ead = np.array([10_000.0, 10_000.0])
    princ = np.array([500.0, 5_500.0])               # a = 0.05 and 0.55
    recoveries = 0.20 * (ead - princ)
    lgd = 100 * (ead - princ - recoveries) / ead
    return pd.DataFrame({
        "EAD": ead, "princ_rec": princ, "interest_rec": [400.0, 2_000.0],
        "monthly_payment": [300.0, 250.0], "Target_Variable_Loss": lgd,
    })


def test_components_recover_the_identity():
    comp = lgd_components(_loans())
    np.testing.assert_allclose(comp["a_repaid"], [0.05, 0.55])
    np.testing.assert_allclose(comp["r_recovered"], [0.20, 0.20])
    np.testing.assert_allclose(comp["lgd"], [76.0, 36.0])
    np.testing.assert_allclose(comp["lgd_at_default"], [80.0, 80.0])
    rebuilt = 100 * (1 - comp["a_repaid"]) * (1 - comp["r_recovered"])
    np.testing.assert_allclose(rebuilt, comp["lgd"])
    np.testing.assert_allclose(comp["months_paid"], [3.0, 30.0])


def test_fully_repaid_loan_has_no_recovery_rate():
    df = _loans()
    df.loc[0, "princ_rec"] = 10_000.0
    df.loc[0, "Target_Variable_Loss"] = 0.0
    comp = lgd_components(df)
    assert np.isnan(comp.loc[0, "r_recovered"])


def test_variance_explained_bounds():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.uniform(size=5_000))
    assert variance_explained(10 * x, x) == pytest.approx(1.0, abs=0.01)
    assert variance_explained(pd.Series(rng.normal(size=5_000)), x) < 0.03


def test_lgd_by_months_paid_counts_every_loan():
    comp = lgd_components(_loans())
    table = lgd_by_months_paid(comp)
    assert table["loans"].sum() == 2
