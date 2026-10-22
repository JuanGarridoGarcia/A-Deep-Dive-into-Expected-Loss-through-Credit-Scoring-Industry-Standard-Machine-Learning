"""
clean_features_LGD.py

Outlier and missing-value treatment for the LGD model features.

LGD_TREATMENTS lists the treatment of every feature the LGD selection can return;
only the features present in the frame are treated. Parameters are fitted on the
defaulted loans of the development sample. FeaturePipeline_LGD reuses the same
specification, so training and portfolio scoring always treat features alike.
"""

import pandas as pd

from feature_treatment import FeatureTreatment

LGD_TREATMENTS = {
    # Balances with long upper tails: cap at p99 (log1p discarded, it inverts the
    # skew because of the mass near zero)
    "total_credit_revolving_bal": [("cap_upper_quantile", 0.99)],
    "max_bal_owed": [("cap_upper_quantile", 0.99)],
    # Heavy right tail: cap at p99, then log1p if still strongly skewed
    "annual_income": [("cap_upper_quantile", 0.99), ("log1p_if_skewed", 2.0)],
    # Utilisation ratios: economic ceiling at 100%
    "used_credit_share": [("cap_upper_value", 100)],
    "bal_to_cred_lim": [("cap_upper_value", 100)],
    # Counts with a discontinuous upper tail: cap at p99
    "num_open_credit_lines": [("cap_upper_quantile", 0.99)],
    "tot_num_credit_lines": [("cap_upper_quantile", 0.99)],
    "num_inq_in_6mths": [("cap_upper_quantile", 0.99)],
    "num_inq_in_12mths": [("cap_upper_quantile", 0.99)],
    # Negative debt-to-income values are data errors
    "dept_paym_income_ratio": [("clip_lower_value", 0)],
    # Lower-tail errors in the credit history start year
    "earliest_cr_line_year": [("cap_lower_quantile", 0.01)],
    "home_ownership_status": [("replace", {"OTHER": "RENT"})],
}


def fit_treatment_LGD(reference: pd.DataFrame, verbose: bool = True) -> FeatureTreatment:
    """Fit the LGD treatment on the reference (defaulted development) population."""
    return FeatureTreatment(LGD_TREATMENTS, verbose=verbose).fit(reference)


def clean_features_LGD(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Apply the LGD treatments to df, with parameters fitted on `reference`
    (default: df itself).
    """
    reference = df if reference is None else reference[df.columns]
    return fit_treatment_LGD(reference).transform(df)
