"""
clean_features_PD.py

Outlier and missing-value treatment for the PD model features.

PD_TREATMENTS lists the treatment of every feature the PD selection can return;
only the features present in the frame are treated. Parameters are fitted on the
development sample (matured loans) and applied unchanged to the whole portfolio.
"""

import pandas as pd

from feature_treatment import FeatureTreatment

PD_TREATMENTS = {
    # Heavy right tail: cap at p99, then log1p if still strongly skewed
    "annual_income": [("cap_upper_quantile", 0.99), ("log1p_if_skewed", 2.0)],
    # Utilisation ratios: economic ceiling at 100%
    "used_credit_share": [("cap_upper_value", 100)],
    "bal_to_cred_lim": [("cap_upper_value", 100)],
    # Counts and balances with long upper tails: cap at p99 (log1p discarded for
    # balances, it inverts the skew because of the mass near zero)
    "num_inq_in_12mths": [("cap_upper_quantile", 0.99)],
    "num_inq_in_6mths": [("cap_upper_quantile", 0.99)],
    "max_bal_owed": [("cap_upper_quantile", 0.99)],
    "total_credit_revolving_bal": [("cap_upper_quantile", 0.99)],
    "num_open_credit_lines": [("cap_upper_quantile", 0.99)],
    "tot_num_credit_lines": [("cap_upper_quantile", 0.99)],
    # Negative debt-to-income values are data errors
    "dept_paym_income_ratio": [("clip_lower_value", 0)],
    # Lower-tail errors in the credit history start year
    "earliest_cr_line_year": [("cap_lower_quantile", 0.01)],
    # OTHER (0.07%) grouped into RENT, the closest economic profile
    "home_ownership_status": [("replace", {"OTHER": "RENT"})],
}


def fit_treatment_PD(reference: pd.DataFrame, verbose: bool = True) -> FeatureTreatment:
    """Fit the PD treatment on the reference (development) population."""
    return FeatureTreatment(PD_TREATMENTS, verbose=verbose).fit(reference)


def clean_features_PD(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Apply the PD treatments to df, with parameters fitted on `reference`
    (default: df itself). Columns without a treatment are left as they are,
    apart from median imputation of numeric NaNs.
    """
    reference = df if reference is None else reference[df.columns]
    return fit_treatment_PD(reference).transform(df)
