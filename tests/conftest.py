"""Shared test setup: make the modules in src/ importable and build toy data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LGD_NUMERIC = [
    "total_credit_revolving_bal", "monthly_payment", "num_open_credit_lines",
    "num_rev_trades_op_in_24mths", "used_credit_share", "interest_rate",
    "annual_income", "emp_length", "dept_paym_income_ratio",
]


@pytest.fixture
def lgd_frame():
    """Toy defaulted-loan frame with the LGD feature set."""
    rng = np.random.default_rng(0)
    n = 500
    return pd.DataFrame({
        "total_credit_revolving_bal": rng.lognormal(9, 1, n),
        "monthly_payment": rng.uniform(50, 1500, n),
        "num_open_credit_lines": rng.integers(1, 40, n),
        "num_rev_trades_op_in_24mths": rng.integers(0, 20, n),
        "used_credit_share": rng.uniform(0, 140, n),  # some above the 100 ceiling
        "interest_rate": rng.uniform(5, 30, n),
        "annual_income": rng.lognormal(11, 1.2, n),
        "emp_length": rng.integers(0, 11, n),
        "dept_paym_income_ratio": rng.uniform(0, 40, n),
        "loan_term_months": rng.choice([36, 60], n),
    })
