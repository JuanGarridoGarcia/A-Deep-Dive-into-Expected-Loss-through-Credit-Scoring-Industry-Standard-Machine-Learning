"""
Feature selection for binary classification with a categorical target variable.

Applied metrics:
  - mutual_info     : Mutual information score (captures non-linear relationships)
  - perm_importance : Permutation importance using a Random Forest
  - lasso_coef      : Absolute coefficient from L1-penalised Logistic Regression
  - boruta_support  : 1 confirmed relevant / 0 rejected / -1 tentative

Discard criterion (discard=1): a feature is dropped only if it falls below
the threshold in ALL THREE main metrics simultaneously.
Boruta is kept as an informational column and does not vote in the
automatic discard decision.

Requirements:
    pip install pandas numpy scikit-learn boruta
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from scipy.stats import chi2_contingency
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from boruta import BorutaPy


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal encoding so that every column can be fed to sklearn estimators.

    - Numeric columns  : left as-is (NaNs are imputed later in each method).
    - Categorical cols : label-encoded.  NaNs are preserved by temporarily
                         converting them to strings, encoding, then restoring NaN.
    """
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        mask_nan = df_enc[col].isna()
        df_enc[col] = df_enc[col].astype(str)
        df_enc[col] = le.fit_transform(df_enc[col])
        df_enc.loc[mask_nan, col] = np.nan
    return df_enc


def _safe_impute(X: np.ndarray) -> np.ndarray:
    """Replace NaNs with the column median so that estimators can run."""
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(X)


# ──────────────────────────────────────────────
# Individual scoring methods
# ──────────────────────────────────────────────

def _mutual_info(X: np.ndarray, y: np.ndarray, feature_names: list) -> pd.Series:
    is_regression = len(np.unique(y)) > 20
    if is_regression:
        mi = mutual_info_regression(X, y, random_state=42)
    else:
        mi = mutual_info_classif(X, y, random_state=42)
    return pd.Series(mi, index=feature_names, name="mutual_info")


def _permutation_importance_score(X: np.ndarray, y: np.ndarray,
                                   feature_names: list,
                                   n_repeats: int = 10,
                                   sample_size: int = 500_000) -> pd.Series:
    is_regression = len(np.unique(y)) > 20
    if len(X) > sample_size:
        stratify = y if not is_regression else None
        X, y = resample(X, y, n_samples=sample_size, stratify=stratify, random_state=42)

    if is_regression:
        rf = RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=42)
    else:
        rf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)

    rf.fit(X, y)
    result = permutation_importance(rf, X, y, n_repeats=n_repeats, random_state=42, n_jobs=1)
    return pd.Series(result.importances_mean, index=feature_names, name="perm_importance")


def _lasso_coef(X: np.ndarray, y: np.ndarray, feature_names: list) -> pd.Series:
    is_regression = len(np.unique(y)) > 20
    if is_regression:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", Lasso(alpha=0.1, max_iter=2000, random_state=42))
        ])
        pipe.fit(X, y)
        coefs = np.abs(pipe.named_steps["clf"].coef_)
    else:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l1", solver="saga", C=0.1,
                                       max_iter=2000, random_state=42))
        ])
        pipe.fit(X, y)
        coefs = np.abs(pipe.named_steps["clf"].coef_[0])
    return pd.Series(coefs, index=feature_names, name="lasso_coef")


def _boruta_score(X: np.ndarray, y: np.ndarray, feature_names: list,
                  max_iter: int = 50) -> pd.Series:
    """
    Boruta algorithm.

    Creates 'shadow features' (column-wise permuted copies of every feature)
    and trains a Random Forest on the combined original + shadow set.
    A feature is confirmed as relevant only if it consistently outperforms
    the best shadow feature across iterations (binomial test).

    Returns
    -------
    1  → confirmed relevant
    0  → rejected
   -1  → tentative (inconclusive — worth manual inspection)
    """
    rf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
    boruta_sel = BorutaPy(
        rf, n_estimators="auto", max_iter=max_iter, random_state=42, verbose=0
    )
    boruta_sel.fit(X, y)

    support = np.where(
        boruta_sel.support_, 1,
        np.where(boruta_sel.support_weak_, -1, 0)
    )
    return pd.Series(support, index=feature_names, name="boruta_support")


# ──────────────────────────────────────────────
# Pre-screening: variables requiring manual review
# ──────────────────────────────────────────────

def diagnose_features(
    df_features: pd.DataFrame,
    nan_threshold: float = 5.0,
    min_category_freq_pct: float = 1.0,
    cardinality_ratio: float = 0.05,
) -> pd.DataFrame:
    """
    Identify features that require manual inspection before running select_features.

    A feature is flagged if it meets one or more of the following criteria:

      NaNs            : percentage of missing values >= nan_threshold.
      High_Cardinality: the least frequent category appears in < min_category_freq_pct
                        of rows, OR the number of unique categories exceeds
                        cardinality_ratio * n_rows.  Numeric columns are never flagged
                        for cardinality.
      Zero_Variance   : a single value accounts for >= 95 % of non-null rows,
                        meaning the column carries almost no information.

    Parameters
    ----------
    df_features           : DataFrame with ONLY the feature columns (no target).
    nan_threshold         : Minimum % of NaNs to flag a feature (default 5.0).
    min_category_freq_pct : Minimum % frequency of the rarest category below which
                            a categorical feature is flagged (default 1.0).
    cardinality_ratio     : If n_unique / n_rows exceeds this ratio the feature is
                            also flagged for high cardinality (default 0.05).

    Returns
    -------
    pd.DataFrame — one row per flagged feature, sorted by reason then pct_NaNs desc.
    Columns:
      feature              : feature name
      dtype                : 'numeric' or 'categorical'
      pct_NaNs             : percentage of missing values
      n_categories         : number of unique categories (NaN for numeric)
      min_category_freq_pct: frequency % of the rarest category (NaN for numeric)
      reason               : 'NaNs' | 'High_Cardinality' | 'Zero_Variance'
                             (comma-separated if multiple reasons apply)
    """

    n_rows = len(df_features)
    records = {}

    for col in df_features.columns:

        reasons = []

        # ── Basic stats ──────────────────────────────────────────────────────
        n_nan      = df_features[col].isna().sum()
        pct_nan    = round(n_nan / n_rows * 100, 2)
        is_cat     = df_features[col].dtype == object or str(df_features[col].dtype) == "category"
        dtype_label = "categorical" if is_cat else "numeric"

        # ── Cardinality stats (categorical only) ─────────────────────────────
        n_categories          = np.nan
        min_freq_pct          = np.nan

        if is_cat:
            value_counts      = df_features[col].dropna().value_counts()
            n_categories      = len(value_counts)
            min_freq_pct      = round(value_counts.min() / n_rows * 100, 4)

            high_card = (
                min_freq_pct < min_category_freq_pct
                or n_categories > cardinality_ratio * n_rows
            )
            if high_card:
                reasons.append("High_Cardinality")

        # ── NaN check ────────────────────────────────────────────────────────
        if pct_nan >= nan_threshold:
            reasons.append("NaNs_Number")

        # ── Zero variance check ──────────────────────────────────────────────
        non_null = df_features[col].dropna()
        if len(non_null) > 0:
            top_freq_pct = non_null.value_counts(normalize=True).iloc[0] * 100
            if top_freq_pct >= 90.0:
                reasons.append("Zero_Variance")

        # ── Record if flagged ─────────────────────────────────────────────────
        if reasons:
            records[col] = {
                "feature":               col,
                "dtype":                 dtype_label,
                "pct_NaNs":              pct_nan,
                "n_categories":          n_categories,
                "min_category_freq_pct": min_freq_pct,
                "top_value_freq_pct":    round(top_freq_pct, 2),
                "reason":                ", ".join(reasons),
            }

    if not records:
        print("✓ No features flagged. All columns look clean for select_features.")
        return pd.DataFrame(columns=[
            "feature", "dtype", "pct_NaNs",
            "n_categories", "min_category_freq_pct", "top_value_freq_pct", "reason"
        ])

    result = pd.DataFrame(records.values())

    # Sort: Zero_Variance first (direct discard), then High_Cardinality, then NaNs
    priority = {"Zero_Variance": 0, "High_Cardinality": 1, "NaNs": 2}
    result["_sort_key"] = result["reason"].apply(
        lambda r: min(priority.get(x.strip(), 3) for x in r.split(","))
    )
    result = (
        result
        .sort_values(["_sort_key", "pct_NaNs"], ascending=[True, False])
        .drop(columns="_sort_key")
        .reset_index(drop=True)
    )

    n_zero_var   = result["reason"].str.contains("Zero_Variance").sum()
    n_high_card  = result["reason"].str.contains("High_Cardinality").sum()
    n_nans       = result["reason"].str.contains("NaNs").sum()

    print(f"⚠ Features flagged for manual review: {len(result)}")
    print(f"  · Zero_Variance   : {n_zero_var}  → safe to discard directly")
    print(f"  · High_Cardinality: {n_high_card}  → requires encoding decision")
    print(f"  · NaNs_Number     : {n_nans}  → requires imputation strategy")

    return result


# ──────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────

def select_features(
    df_features: pd.DataFrame,
    y: pd.Series,
    mi_threshold: float = 0.01,
    perm_threshold: float = 0.0,
    lasso_threshold: float = 1e-4,
    boruta_max_iter: int = 50,
    perm_n_repeats: int = 10,
    run_boruta: bool = True,
    sample_size: int = 500_000,
) -> pd.DataFrame:
    """
    Assess the predictive power of every column in `df_features` against `y`.

    Parameters
    ----------
    df_features      : DataFrame containing ONLY the feature columns (no target).
    y                : Series with the binary target variable (categorical/string).
    mi_threshold     : Minimum mutual information score to keep a feature.
    perm_threshold   : Minimum permutation importance to keep a feature.
    lasso_threshold  : Minimum absolute L1 coefficient to keep a feature.
    boruta_max_iter  : Maximum iterations for Boruta.
    perm_n_repeats   : Number of shuffling repetitions for permutation importance.
    run_boruta       : Set to False to skip Boruta and get a faster result.
    sample_size      : Number of rows used for permutation importance and Boruta.
                       Reduces memory usage on large datasets (default 500_000).

    Returns
    -------
    pd.DataFrame with columns:
      feature         : feature name
      discard         : 1 = recommended for removal, 0 = keep
      mutual_info     : mutual information score
      perm_importance : mean permutation importance
      lasso_coef      : absolute L1 coefficient
      boruta_support  : 1 confirmed / 0 rejected / -1 tentative / NaN if skipped

    Discard logic
    -------------
    A feature is flagged (discard=1) only when it falls below the threshold
    in ALL THREE main metrics.  This conservative AND-rule avoids discarding
    features that one method might miss but another catches.
    Boruta is informational only and does not affect the discard flag.
    """

    # 1. Encode and prepare
    
    print("→ Encoding features...")
    df_enc = _encode_features(df_features)
    feature_names = df_enc.columns.tolist()

    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y.astype(str))

    # Sample BEFORE imputing to avoid memory and speed issues
    if len(df_enc) > sample_size:
        print(f"→ Sampling {sample_size:,} rows from {len(df_enc):,}...")
        stratify = y_enc if len(np.unique(y_enc)) < 20 else None
        idx = resample(
            np.arange(len(df_enc)),
            n_samples=sample_size,
            stratify=stratify,
            random_state=42
        )
        df_enc = df_enc.iloc[idx]
        y_enc  = y_enc[idx]

    X_imp = _safe_impute(df_enc.values)

    # 2. Compute metrics
    print("→ Computing mutual information...")
    mi = _mutual_info(X_imp, y_enc, feature_names)

    print("→ Computing permutation importance (may take a few minutes)...")
    perm = _permutation_importance_score(
        X_imp, y_enc, feature_names,
        n_repeats=perm_n_repeats,
        sample_size=sample_size
    )

    print("→ Computing L1 (Lasso) coefficients...")
    lasso = _lasso_coef(X_imp, y_enc, feature_names)

    boruta_series = pd.Series(np.nan, index=feature_names, name="boruta_support")
    is_regression = len(np.unique(y_enc)) > 20
    if run_boruta and not is_regression:
        print("→ Running Boruta (may take several minutes)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boruta_series = _boruta_score(X_imp, y_enc, feature_names, max_iter=boruta_max_iter)
    elif run_boruta and is_regression:
        print("→ Skipping Boruta (not supported for regression targets).")

    # 3. Assemble results
    results = pd.DataFrame({
        "feature":         feature_names,
        "mutual_info":     mi.values,
        "perm_importance": perm.values,
        "lasso_coef":      lasso.values,
        "boruta_support":  boruta_series.values,
    })

    # 4. Apply discard criterion
    below_mi    = results["mutual_info"]     < mi_threshold
    below_perm  = results["perm_importance"] <= perm_threshold
    below_lasso = results["lasso_coef"]      < lasso_threshold

    results["discard"] = (below_mi & below_perm & below_lasso).astype(int)

    # 5. Sort: kept features first (by mutual information desc), discarded last
    results = results.sort_values(
        ["discard", "mutual_info"], ascending=[True, False]
    ).reset_index(drop=True)

    n_discard = results["discard"].sum()
    n_keep    = len(results) - n_discard
    print(f"\n✓ Done.  Features kept: {n_keep} | Discarded: {n_discard}")

    return results


# ──────────────────────────────────────────────
# Individual correlation methods
# ──────────────────────────────────────────────

def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér entre dos variables categóricas."""
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion, correction=False)[0]
    n = confusion.sum().sum()
    k = min(confusion.shape) - 1
    return np.sqrt(chi2 / (n * k)) if k > 0 else 0.0


def _eta_squared(numeric: pd.Series, categorical: pd.Series) -> float:
    """Eta cuadrado entre una variable numérica y una categórica."""
    groups = [numeric[categorical == cat].dropna()
              for cat in categorical.unique()]
    grand_mean = numeric.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = ((numeric - grand_mean) ** 2).sum()
    return ss_between / ss_total if ss_total > 0 else 0.0


# ──────────────────────────────────────────────
# Combined correlation matrix
# ──────────────────────────────────────────────

from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
    is_categorical_dtype
)

def compute_plot_correlation_matrix(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """
    Builds a unified correlation matrix combining:
      - Pearson     for numeric  vs numeric
      - Cramér's V  for categorical vs categorical
      - Eta squared for numeric vs categorical

    All metrics are in [0, 1] (absolute association strength).
    Also plots the correlation heatmap.
    """
    cols    = df.columns.tolist()
    n       = len(cols)
    matrix  = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)
    is_cat = {
        col: (
            is_object_dtype(df[col]) or
            is_string_dtype(df[col]) or
            is_categorical_dtype(df[col])
        )
        for col in cols
    }

    for i in range(n):
        for j in range(i + 1, n):
            a, b   = cols[i], cols[j]
            cat_a  = is_cat[a]
            cat_b  = is_cat[b]

            mask   = df[[a, b]].notna().all(axis=1)
            sa, sb = df.loc[mask, a], df.loc[mask, b]

            if not cat_a and not cat_b:
                val = abs(sa.corr(sb))
            elif cat_a and cat_b:
                val = _cramers_v(sa, sb)
            elif not cat_a and cat_b:
                val = _eta_squared(sa, sb)
            else:
                val = _eta_squared(sb, sa)

            matrix.loc[a, b] = val
            matrix.loc[b, a] = val

    fig, ax = plt.subplots(figsize=(20, 16))
    mask_triu = np.triu(np.ones_like(matrix, dtype=bool))

    sns.heatmap(
        matrix,
        mask=mask_triu,
        annot=False,
        cmap="coolwarm",
        vmin=0, vmax=1,
        linewidths=0.3,
        ax=ax
    )
    ax.set_title("Combined Correlation Matrix\n(Pearson | Cramér's V | Eta²)", fontsize=14)
    plt.tight_layout()
    plt.show()

    return matrix

# ──────────────────────────────────────────────
# Heatmap
# ──────────────────────────────────────────────

def analyze_correlated_pairs(matrix: pd.DataFrame, report: pd.DataFrame, threshold: float = 0.75):
    """
    Identifies pairs of features with correlation above the threshold and recommends
    which one to discard based on mutual information (greedy algorithm ordered by
    mutual information descending — the feature with lower MI is discarded).

    Parameters
    ----------
    matrix    : correlation matrix from compute_plot_correlation_matrix
    report    : output from select_features (must contain 'feature' and 'mutual_info')
    threshold : correlation threshold above which a pair is flagged (default 0.75)

    Returns
    -------
    pairs_df  : DataFrame with columns feature_a, feature_b, correlation,
                mi_feature_a, mi_feature_b, discard
    to_drop   : list of features recommended for removal
    """

    mi_lookup = report.set_index("feature")["mutual_info"].to_dict()

    # Identify correlated pairs
    cols  = matrix.columns.tolist()
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = matrix.iloc[i, j]
            if val >= threshold:
                a, b = cols[i], cols[j]
                pairs.append({
                    "feature_a":    a,
                    "feature_b":    b,
                    "correlation":  round(val, 4),
                    "mi_feature_a": round(mi_lookup.get(a, np.nan), 4),
                    "mi_feature_b": round(mi_lookup.get(b, np.nan), 4),
                })

    if not pairs:
        print(f"✓ No pairs above threshold ({threshold}).")
        return pd.DataFrame(columns=[
            "feature_a", "feature_b", "correlation",
            "mi_feature_a", "mi_feature_b", "discard"
        ]), []

    pairs_df = pd.DataFrame(pairs).sort_values("correlation", ascending=False).reset_index(drop=True)

    # Greedy algorithm: order features by MI descending, keep the most informative
    mi_ordered = report.sort_values("mutual_info", ascending=False)["feature"].tolist()
    kept    = []
    to_drop = []

    for feature in mi_ordered:
        correlated_with_kept = pairs_df[
            ((pairs_df["feature_a"] == feature) & (pairs_df["feature_b"].isin(kept))) |
            ((pairs_df["feature_b"] == feature) & (pairs_df["feature_a"].isin(kept)))
        ]
        if correlated_with_kept.empty:
            kept.append(feature)
        else:
            to_drop.append(feature)

    # Add discard column
    pairs_df["discard"] = pairs_df.apply(
        lambda row: row["feature_a"] if row["feature_a"] in to_drop
                    else row["feature_b"] if row["feature_b"] in to_drop
                    else "none",
        axis=1
    )

    print(f"⚠ Pairs above threshold ({threshold}): {len(pairs_df)}")
    print(f"✓ Features recommended for removal: {len(to_drop)}")
    print(f"  {to_drop}")

    return pairs_df, to_drop

def exploratory_feature_selection(
    df: pd.DataFrame,
    features: list,
    target: str,
    model_type: str = 'classification',
    sample_size: int = 200_000,
    test_size: float = 0.2,
    cumulative_gain_threshold: float = 0.80,
    max_auc_degradation: float = 0.01,
    max_rmse_degradation_pct: float = 0.05,
    shap_sample_size: int = 50_000,
):
    """
    Trains an exploratory XGBoost model to rank features by predictive importance
    and determine a cutoff that retains the most relevant features while preserving
    model performance.

    Two parallel cutoff criteria are computed and compared:
      - Gain-based: cumulative normalized gain + elbow method.
      - SHAP-based: cumulative mean(|SHAP|) + elbow method.

    The SHAP-based cutoff is used as the primary selection criterion due to its
    consistency property and absence of cardinality bias. The gain-based cutoff
    is retained for comparison and portfolio documentation.

    Parameters
    ----------
    df                        : Master dataframe.
    features                  : List of feature column names.
    target                    : Name of the target column.
    model_type                : 'classification' for PD, 'regression' for LGD.
    sample_size               : Number of rows to sample for training.
    test_size                 : Proportion of data used for evaluation.
    cumulative_gain_threshold : Cumulative importance threshold for cutoff (default 0.80).
                                Applied to both gain and SHAP criteria.
    max_auc_degradation       : Maximum acceptable AUC drop for classification (default 0.01).
    max_rmse_degradation_pct  : Maximum acceptable RMSE increase % for regression (default 0.05).
    shap_sample_size          : Number of rows used to compute SHAP values (default 50_000).
                                SHAP is computed on a subsample of X_test for efficiency.

    Returns
    -------
    importance_df     : DataFrame with weight, gain, cover, shap_mean_abs per feature,
                        sorted by shap_mean_abs descending.
    selected_features : List of features selected after the SHAP-based cutoff and
                        performance validation.
    metrics           : Dict with full model and reduced model performance metrics.
    """

    is_classification = model_type == 'classification'

    # ── 1. Prepare data ───────────────────────────────────────────────────────
    print("→ Preparing data...")
    df_model = df[features + [target]].copy()

    for col in df_model[features].select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        mask_nan = df_model[col].isna()
        df_model[col] = df_model[col].astype(str)
        df_model[col] = le.fit_transform(df_model[col])
        df_model.loc[mask_nan, col] = np.nan

    if len(df_model) > sample_size:
        print(f"→ Sampling {sample_size:,} rows from {len(df_model):,}...")
        # Stratified sample to preserve class distribution before the train/test split.
        # Consistent with the stratified split applied below.
        if is_classification:
            classes = df_model[target].unique()
            samples = []
            for c in classes:
                n = int(sample_size * (df_model[target] == c).sum() / len(df_model))
                samples.append(df_model[df_model[target] == c].sample(n=n, random_state=42))
            df_model = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            df_model = df_model.sample(n=sample_size, random_state=42)

    y = df_model[target].values

    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(
        imputer.fit_transform(df_model[features]),
        columns=features
    )

    stratify = y if is_classification else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    # ── 2. Train full model ───────────────────────────────────────────────────
    print("→ Training full exploratory model...")
    if is_classification:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42, n_jobs=-1, eval_metric='auc',
            verbosity=0
        )
    else:
        model = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
            verbosity=0
        )

    model.fit(X_train, y_train)

    if is_classification:
        y_prob      = model.predict_proba(X_test)[:, 1]
        y_pred_full = model.predict(X_test)
        full_auc       = roc_auc_score(y_test, y_prob)
        full_accuracy  = accuracy_score(y_test, y_pred_full)
        full_precision = precision_score(y_test, y_pred_full)
        full_recall    = recall_score(y_test, y_pred_full)
        full_f1        = f1_score(y_test, y_pred_full)
        print(f"→ Full model — AUC: {full_auc:.4f} | Accuracy: {full_accuracy:.4f} | "
              f"Precision: {full_precision:.4f} | Recall: {full_recall:.4f} | F1: {full_f1:.4f}")
    else:
        y_pred    = model.predict(X_test)
        full_rmse = mean_squared_error(y_test, y_pred) ** 0.5
        full_mae  = mean_absolute_error(y_test, y_pred)
        full_r2   = r2_score(y_test, y_pred)
        print(f"→ Full model — RMSE: {full_rmse:.4f} | MAE: {full_mae:.4f} | R²: {full_r2:.4f}")

    # ── 3. Gain-based feature importance ─────────────────────────────────────
    print("→ Computing gain-based feature importance...")
    booster = model.get_booster()

    weight_raw = booster.get_score(importance_type='weight')
    gain_raw   = booster.get_score(importance_type='gain')
    cover_raw  = booster.get_score(importance_type='cover')

    # Raw values are kept for gain and cover. Normalizing by sum gives proportions;
    # normalizing by max (previous version) distorted the cumulative gain calculation.
    total_gain  = sum(gain_raw.values())  if gain_raw  else 1
    total_cover = sum(cover_raw.values()) if cover_raw else 1
    total_weight = sum(weight_raw.values()) if weight_raw else 1

    importance_df = pd.DataFrame({
        'feature': features,
        'weight':  [round(weight_raw.get(f, 0) / total_weight, 6) for f in features],
        'gain':    [round(gain_raw.get(f, 0)   / total_gain,   6) for f in features],
        'cover':   [round(cover_raw.get(f, 0)  / total_cover,  6) for f in features],
    })

    # ── 4. SHAP-based feature importance ──────────────────────────────────────
    # TreeSHAP (Lundberg et al., 2018) is exact for tree ensembles and runs in
    # O(T·L·D²) — fast even on large datasets. We compute on a subsample of
    # X_test for efficiency; 50k rows is sufficient for a stable mean(|SHAP|).
    print(f"→ Computing SHAP values (sample: {min(shap_sample_size, len(X_test)):,} rows)...")
    X_shap = X_test.sample(
        n=min(shap_sample_size, len(X_test)),
        random_state=42
    )

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # For classifiers, shap_values may be a list [class_0, class_1]; take class 1.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_mean_abs = np.abs(shap_values).mean(axis=0)
    shap_total    = shap_mean_abs.sum() if shap_mean_abs.sum() > 0 else 1

    importance_df['shap_mean_abs'] = [
        round(shap_mean_abs[X_shap.columns.get_loc(f)] / shap_total, 6)
        for f in features
    ]

    # Sort by SHAP (primary criterion) descending.
    importance_df = importance_df.sort_values('shap_mean_abs', ascending=False).reset_index(drop=True)

    # ── 5. Determine cutoffs ──────────────────────────────────────────────────
    def _compute_cutoff(values: np.ndarray, threshold: float) -> int:
        """
        Combines cumulative threshold and elbow method to determine feature cutoff.
        Takes the more conservative (larger) of the two.

        The elbow is detected via the second derivative of the sorted importance
        curve. Note: this method is sensitive to noise when consecutive values are
        similar. It is used as a lower bound, not as the sole criterion.
        """
        norm       = values / values.sum()
        cumulative = np.cumsum(norm)

        cutoff_cumulative = int(np.searchsorted(cumulative, threshold)) + 1

        if len(values) > 2:
            second_deriv  = np.diff(np.diff(values))
            cutoff_elbow  = int(np.argmax(np.abs(second_deriv))) + 2
        else:
            cutoff_elbow = len(values)

        return max(cutoff_cumulative, cutoff_elbow)

    gain_values  = importance_df['gain'].values
    shap_values_ = importance_df['shap_mean_abs'].values

    cutoff_gain = _compute_cutoff(gain_values,  cumulative_gain_threshold)
    cutoff_shap = _compute_cutoff(shap_values_, cumulative_gain_threshold)

    gain_sorted = importance_df.sort_values('gain', ascending=False).reset_index(drop=True)
    cumulative_gain = np.cumsum(gain_sorted['gain'].values)
    cutoff_gain = _compute_cutoff(gain_sorted['gain'].values, cumulative_gain_threshold)

    print(f"→ Gain-based cutoff:  {cutoff_gain} features")
    print(f"→ SHAP-based cutoff:  {cutoff_shap} features (primary criterion)")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 6a. Gain bar chart
    colors_gain = ['#185FA5' if i < cutoff_gain else '#B5D4F4'
                for i in range(len(gain_sorted))]
    axes[0, 0].bar(range(len(gain_sorted)), gain_sorted['gain'], color=colors_gain)
    axes[0, 0].axvline(x=cutoff_gain - 0.5, color='#A32D2D', linestyle='--',
                    linewidth=1.5, label=f'Cutoff ({cutoff_gain} features)')
    axes[0, 0].set_xlabel('Feature rank', fontsize=11)
    axes[0, 0].set_ylabel('Normalized gain (proportion)', fontsize=11)
    axes[0, 0].set_title('Feature importance — Gain', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.2, axis='y')

    # 6b. Gain cumulative curve
    axes[0, 1].plot(range(1, len(gain_sorted) + 1), cumulative_gain,
                    color='#185FA5', linewidth=2)
    axes[0, 1].axhline(y=cumulative_gain_threshold, color='#A32D2D', linestyle='--',
                    linewidth=1, label=f'{int(cumulative_gain_threshold*100)}% threshold')
    axes[0, 1].axvline(x=cutoff_gain, color='#3B6D11', linestyle='--',
                    linewidth=1.5, label=f'Cutoff ({cutoff_gain} features)')
    axes[0, 1].set_xlabel('Number of features', fontsize=11)
    axes[0, 1].set_ylabel('Cumulative gain', fontsize=11)
    axes[0, 1].set_title('Cumulative gain curve', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.2)

    # 6c. SHAP bar chart
    colors_shap = ['#185FA5' if i < cutoff_shap else '#B5D4F4'
                   for i in range(len(importance_df))]
    axes[1, 0].bar(range(len(importance_df)), importance_df['shap_mean_abs'],
                   color=colors_shap)
    axes[1, 0].axvline(x=cutoff_shap - 0.5, color='#A32D2D', linestyle='--',
                       linewidth=1.5, label=f'Cutoff ({cutoff_shap} features)')
    axes[1, 0].set_xlabel('Feature rank', fontsize=11)
    axes[1, 0].set_ylabel('mean(|SHAP|) — normalized', fontsize=11)
    axes[1, 0].set_title('Feature importance — SHAP', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.2, axis='y')

    # 6d. SHAP cumulative curve
    cumulative_shap = np.cumsum(importance_df['shap_mean_abs'].values)
    axes[1, 1].plot(range(1, len(importance_df) + 1), cumulative_shap,
                    color='#185FA5', linewidth=2)
    axes[1, 1].axhline(y=cumulative_gain_threshold, color='#A32D2D', linestyle='--',
                       linewidth=1, label=f'{int(cumulative_gain_threshold*100)}% threshold')
    axes[1, 1].axvline(x=cutoff_shap, color='#3B6D11', linestyle='--',
                       linewidth=1.5, label=f'Cutoff ({cutoff_shap} features)')
    axes[1, 1].set_xlabel('Number of features', fontsize=11)
    axes[1, 1].set_ylabel('Cumulative mean(|SHAP|)', fontsize=11)
    axes[1, 1].set_title('Cumulative SHAP curve', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.2)

    plt.suptitle('Exploratory feature selection — Gain vs SHAP', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    # SHAP summary plot (beeswarm) — feature impact direction and magnitude
    print("→ Generating SHAP summary plot...")
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=features,
        max_display=min(cutoff_shap, len(features)),
        show=True
    )

    # ── 7. Validate reduced model (SHAP-based cutoff) ─────────────────────────
    # The reduced model is validated against the full model performance.
    # If degradation exceeds the threshold, cutoff is expanded by 1 until it passes.
    print("\n→ Validating reduced model (SHAP-based cutoff)...")
    cutoff = cutoff_shap

    while cutoff <= len(features):
        selected_features = importance_df['feature'].iloc[:cutoff].tolist()

        X_train_red = X_train[selected_features]
        X_test_red  = X_test[selected_features]

        if is_classification:
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            model_red = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42, n_jobs=-1, verbosity=0
            )
            model_red.fit(X_train_red, y_train)
            y_prob_red        = model_red.predict_proba(X_test_red)[:, 1]
            y_pred_red        = model_red.predict(X_test_red)
            reduced_auc       = roc_auc_score(y_test, y_prob_red)
            reduced_accuracy  = accuracy_score(y_test, y_pred_red)
            reduced_precision = precision_score(y_test, y_pred_red)
            reduced_recall    = recall_score(y_test, y_pred_red)
            reduced_f1        = f1_score(y_test, y_pred_red)
            degradation       = full_auc - reduced_auc
            print(f"  [{cutoff} features] AUC: {reduced_auc:.4f} | "
                  f"Accuracy: {reduced_accuracy:.4f} | Precision: {reduced_precision:.4f} | "
                  f"Recall: {reduced_recall:.4f} | F1: {reduced_f1:.4f} | "
                  f"Degradation: {degradation:.4f}")
            if degradation <= max_auc_degradation:
                break
        else:
            model_red = XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0
            )
            model_red.fit(X_train_red, y_train)
            y_pred_red   = model_red.predict(X_test_red)
            reduced_rmse = mean_squared_error(y_test, y_pred_red) ** 0.5
            reduced_mae  = mean_absolute_error(y_test, y_pred_red)
            reduced_r2   = r2_score(y_test, y_pred_red)
            degradation  = (reduced_rmse - full_rmse) / full_rmse
            print(f"  [{cutoff} features] RMSE: {reduced_rmse:.4f} | "
                  f"MAE: {reduced_mae:.4f} | R²: {reduced_r2:.4f} | "
                  f"Degradation: {degradation:.2%}")
            if degradation <= max_rmse_degradation_pct:
                break

        cutoff += 1

    # ── 8. Assemble metrics ───────────────────────────────────────────────────
    if is_classification:
        metrics = {
            'full_model': {
                'auc':       round(full_auc, 4),
                'accuracy':  round(full_accuracy, 4),
                'precision': round(full_precision, 4),
                'recall':    round(full_recall, 4),
                'f1':        round(full_f1, 4),
            },
            'reduced_model': {
                'n_features': cutoff,
                'auc':        round(reduced_auc, 4),
                'accuracy':   round(reduced_accuracy, 4),
                'precision':  round(reduced_precision, 4),
                'recall':     round(reduced_recall, 4),
                'f1':         round(reduced_f1, 4),
                'auc_degradation': round(full_auc - reduced_auc, 4),
            },
            'cutoffs': {
                'gain_based': cutoff_gain,
                'shap_based': cutoff_shap,
                'final':      cutoff,
            }
        }
    else:
        metrics = {
            'full_model': {
                'rmse': round(full_rmse, 4),
                'mae':  round(full_mae, 4),
                'r2':   round(full_r2, 4),
            },
            'reduced_model': {
                'n_features':       cutoff,
                'rmse':             round(reduced_rmse, 4),
                'mae':              round(reduced_mae, 4),
                'r2':               round(reduced_r2, 4),
                'rmse_degradation': f"{degradation:.2%}",
            },
            'cutoffs': {
                'gain_based': cutoff_gain,
                'shap_based': cutoff_shap,
                'final':      cutoff,
            }
        }

    print(f"\n✓ Done. Selected {len(selected_features)} features (SHAP-based cutoff).")
    if cutoff_gain != cutoff_shap:
        print(f"  Note: gain cutoff ({cutoff_gain}) and SHAP cutoff ({cutoff_shap}) diverged. "
              f"Review importance_df to understand which features differ between criteria.")

    return importance_df, selected_features, metrics
