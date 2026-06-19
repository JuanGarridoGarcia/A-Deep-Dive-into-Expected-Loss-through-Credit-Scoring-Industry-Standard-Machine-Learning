from scipy import stats
import pandas as pd
import numpy as np

def compare_distributions(df, feature, target, alpha=0.05):

    grupos = df[feature].dropna().unique()
    assert len(grupos) == 2, "La función está diseñada para variables con exactamente 2 categorías."

    g1 = df.loc[df[feature] == grupos[0], target].dropna()
    g2 = df.loc[df[feature] == grupos[1], target].dropna()

    stat, p_value = stats.mannwhitneyu(g1, g2, alternative='two-sided')

    # Cohen's d
    pooled_std = ((g1.std() ** 2 + g2.std() ** 2) / 2) ** 0.5
    cohens_d = abs(g1.mean() - g2.mean()) / pooled_std

    if cohens_d < 0.2:
        efecto = "negligible"
    elif cohens_d < 0.5:
        efecto = "small"
    elif cohens_d < 0.8:
        efecto = "moderate"
    else:
        efecto = "big"

    print(f"Compared groups: {grupos[0]} vs {grupos[1]}")
    print(f"Means:            {g1.mean():.2f} vs {g2.mean():.2f}")
    print(f"Medians:          {g1.median():.2f} vs {g2.median():.2f}")
    print(f"Mann-Whitney U:    {stat:.0f}")
    print(f"p-value:           {p_value:.4f} {'→ significative difference' if p_value < alpha else '→ NOT significative difference'}")
    print(f"Cohen's d:         {cohens_d:.3f} → {efecto} effect")

def dataset_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single-pass dataset audit. Returns one row per feature with:
    dtype, n_unique, nulls (count + %), min, max, mean, std, skew.
    Handles numeric, categorical, object and boolean columns.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(s)

        row = {
            'feature':    col,
            'dtype':      str(s.dtype),
            'n_unique':   s.nunique(),
            'null_count': s.isna().sum(),
            'null_pct':   round(s.isna().mean() * 100, 2),
            'min':        round(s.min(), 4)  if is_numeric else None,
            'max':        round(s.max(), 4)  if is_numeric else None,
            'mean':       round(s.mean(), 4) if is_numeric else None,
            'std':        round(s.std(), 4)  if is_numeric else None,
            'skew':       round(s.skew(), 4) if is_numeric else None,
        }
        rows.append(row)

    return pd.DataFrame(rows).set_index('feature')

def outlier_summary(df, features):
    rows = []
    for col in features:
        s = df[col].dropna()
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        
        moderate = ((s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)).sum()
        extreme  = ((s < Q1 - 3.0 * IQR) | (s > Q3 + 3.0 * IQR)).sum()
        
        rows.append({
            'feature':       col,
            'Q1':            round(Q1, 2),
            'Q3':            round(Q3, 2),
            'IQR':           round(IQR, 2),
            'lower_1.5':    round(Q1 - 1.5 * IQR, 2),
            'upper_1.5':    round(Q3 + 1.5 * IQR, 2),
            'lower_3.0':    round(Q1 - 3.0 * IQR, 2),
            'upper_3.0':    round(Q3 + 3.0 * IQR, 2),
            'moderate_n':    moderate,
            'moderate_pct':  round(moderate / len(s) * 100, 2),
            'extreme_n':     extreme,
            'extreme_pct':   round(extreme / len(s) * 100, 2),
        })
    
    return pd.DataFrame(rows).set_index('feature')

def clean_features_PD(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature treatments defined for the PD model.
    Treatments are based on parameters computed on the training population
    and must be applied consistently at inference time via FeaturePipeline_PD.

    Treatments applied:
        - annual_income      : cap at p99, log1p
        - used_credit_share  : cap at 100 (economic ceiling)
        - bal_to_cred_lim    : cap at 100, impute NaNs with median post-capping
        - num_inq_in_12mths  : cap at p99 (= 11)
        - max_bal_owed       : log1p
        - dept_paym_income_ratio : replace negatives with 0
        - earliest_cr_line_year  : cap at p1 (lower tail errors)
    """
    df = df.copy()

    # annual_income — cap at p99, then check skew before applying log1p
    cap_annual_income = df['annual_income'].quantile(0.99)
    df['annual_income'] = df['annual_income'].clip(upper=cap_annual_income)

    skew_post_cap = df['annual_income'].skew()
    print(f"annual_income skew post-capping: {skew_post_cap:.4f}")

    if skew_post_cap > 2:
        df['annual_income'] = np.log1p(df['annual_income'])
        print("log1p applied")
    else:
        print("skew acceptable post-capping, log1p not applied")

    # used_credit_share — economic ceiling at 100
    df['used_credit_share'] = df['used_credit_share'].clip(upper=100)

    # bal_to_cred_lim — economic ceiling at 100, then impute NaNs with post-capping median
    df['bal_to_cred_lim'] = df['bal_to_cred_lim'].clip(upper=100)
    median_bal_to_cred_lim = df['bal_to_cred_lim'].median()
    df['bal_to_cred_lim'] = df['bal_to_cred_lim'].fillna(median_bal_to_cred_lim)

    # num_inq_in_12mths — cap at p99
    cap_num_inq_in_12mths = df['num_inq_in_12mths'].quantile(0.99)
    df['num_inq_in_12mths'] = df['num_inq_in_12mths'].clip(upper=cap_num_inq_in_12mths)

    # max_bal_owed — cap at p99 for consistency with annual_income treatment
    # log1p discarded: distribution shape makes transformation counterproductive
    cap_max_bal_owed = df['max_bal_owed'].quantile(0.99)
    df['max_bal_owed'] = df['max_bal_owed'].clip(upper=cap_max_bal_owed)

    # dept_paym_income_ratio — replace negatives with 0
    df['dept_paym_income_ratio'] = df['dept_paym_income_ratio'].clip(lower=0)

    # earliest_cr_line_year — cap at p1 (lower tail errors)
    cap_earliest_cr_line_year = df['earliest_cr_line_year'].quantile(0.01)
    df['earliest_cr_line_year'] = df['earliest_cr_line_year'].clip(lower=cap_earliest_cr_line_year)

    return df

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
