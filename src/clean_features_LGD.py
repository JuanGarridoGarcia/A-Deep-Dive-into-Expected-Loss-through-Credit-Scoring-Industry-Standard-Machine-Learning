"""
clean_features_LGD.py

Outlier treatment for the LGD model features (defaulted loan population).
The same decisions are applied at inference time by FeaturePipeline_LGD.
"""

import numpy as np
import pandas as pd


def clean_features_LGD(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature treatments defined for the LGD model.
    Parameters are computed on the defaulted training population and must be
    applied consistently at inference time via FeaturePipeline_LGD.

    Treatments applied:
        - total_credit_revolving_bal : cap at p99, log1p if skew > 2 post-capping
        - annual_income              : cap at p99, log1p if skew > 2 post-capping
        - used_credit_share          : cap at 100 (economic ceiling)
        - num_open_credit_lines      : cap at p99 (discontinuous upper tail)
    """
    df = df.copy()

    # total_credit_revolving_bal — cap at p99, no log1p
    # log1p discarded: same pathology as max_bal_owed in PD model —
    # transformation inverts skew due to distribution shape post-capping
    cap_total_credit_revolving_bal = df['total_credit_revolving_bal'].quantile(0.99)
    df['total_credit_revolving_bal'] = df['total_credit_revolving_bal'].clip(upper=cap_total_credit_revolving_bal)

    # annual_income — cap at p99, log1p if skew > 2 post-capping
    cap_annual_income = df['annual_income'].quantile(0.99)
    df['annual_income'] = df['annual_income'].clip(upper=cap_annual_income)

    skew_post_cap = df['annual_income'].skew()
    print(f"annual_income skew post-capping: {skew_post_cap:.4f}")
    if skew_post_cap > 2:
        df['annual_income'] = np.log1p(df['annual_income'])
        print("  log1p applied to annual_income")
    else:
        print("  skew acceptable, log1p not applied to annual_income")

    # used_credit_share — economic ceiling at 100
    df['used_credit_share'] = df['used_credit_share'].clip(upper=100)

    # num_open_credit_lines — cap at p99 (discontinuous upper tail)
    cap_num_open_credit_lines = df['num_open_credit_lines'].quantile(0.99)
    df['num_open_credit_lines'] = df['num_open_credit_lines'].clip(upper=cap_num_open_credit_lines)

    return df
