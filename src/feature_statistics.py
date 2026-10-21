from scipy import stats
import pandas as pd

def compare_distributions(df, feature, target, alpha=0.05):

    groups = df[feature].dropna().unique()
    assert len(groups) == 2, "The function expects a feature with exactly 2 categories."

    g1 = df.loc[df[feature] == groups[0], target].dropna()
    g2 = df.loc[df[feature] == groups[1], target].dropna()

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

    print(f"Compared groups: {groups[0]} vs {groups[1]}")
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


# The cleaning functions now live in their own modules; re-exported here so the
# original exploratory notebook keeps working.
from clean_features_LGD import clean_features_LGD  # noqa: E402,F401
from clean_features_PD import clean_features_PD  # noqa: E402,F401
