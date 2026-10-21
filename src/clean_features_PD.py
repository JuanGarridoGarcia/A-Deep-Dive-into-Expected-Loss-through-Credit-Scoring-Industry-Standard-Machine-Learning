"""
clean_features_PD.py

Outlier treatment for the PD model features (full loan population).
"""

import numpy as np
import pandas as pd


def clean_features_PD(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature treatments defined for the PD model.
    Treatments are based on parameters computed on the training population
    and must be applied consistently at inference time.

    Treatments applied:
        - annual_income      : cap at p99, log1p
        - used_credit_share  : cap at 100 (economic ceiling)
        - bal_to_cred_lim    : cap at 100, impute NaNs with median post-capping
        - num_inq_in_12mths  : cap at p99 (= 11)
        - max_bal_owed       : cap at p99 (log1p discarded, see below)
        - dept_paym_income_ratio : replace negatives with 0
        - earliest_cr_line_year  : cap at p1 (lower tail errors)
    """
    df = df.copy()

    # annual_income: cap at p99, then check skew before applying log1p
    cap_annual_income = df['annual_income'].quantile(0.99)
    df['annual_income'] = df['annual_income'].clip(upper=cap_annual_income)

    skew_post_cap = df['annual_income'].skew()
    print(f"annual_income skew post-capping: {skew_post_cap:.4f}")

    if skew_post_cap > 2:
        df['annual_income'] = np.log1p(df['annual_income'])
        print("log1p applied")
    else:
        print("skew acceptable post-capping, log1p not applied")

    # used_credit_share: economic ceiling at 100
    df['used_credit_share'] = df['used_credit_share'].clip(upper=100)

    # bal_to_cred_lim: economic ceiling at 100, then impute NaNs with post-capping median
    df['bal_to_cred_lim'] = df['bal_to_cred_lim'].clip(upper=100)
    median_bal_to_cred_lim = df['bal_to_cred_lim'].median()
    df['bal_to_cred_lim'] = df['bal_to_cred_lim'].fillna(median_bal_to_cred_lim)

    # num_inq_in_12mths: cap at p99
    cap_num_inq_in_12mths = df['num_inq_in_12mths'].quantile(0.99)
    df['num_inq_in_12mths'] = df['num_inq_in_12mths'].clip(upper=cap_num_inq_in_12mths)

    # max_bal_owed: cap at p99 for consistency with annual_income treatment
    # log1p discarded: distribution shape makes transformation counterproductive
    cap_max_bal_owed = df['max_bal_owed'].quantile(0.99)
    df['max_bal_owed'] = df['max_bal_owed'].clip(upper=cap_max_bal_owed)

    # dept_paym_income_ratio: replace negatives with 0
    df['dept_paym_income_ratio'] = df['dept_paym_income_ratio'].clip(lower=0)

    # earliest_cr_line_year: cap at p1 (lower tail errors)
    cap_earliest_cr_line_year = df['earliest_cr_line_year'].quantile(0.01)
    df['earliest_cr_line_year'] = df['earliest_cr_line_year'].clip(lower=cap_earliest_cr_line_year)

    return df
