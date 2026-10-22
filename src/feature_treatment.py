"""
feature_treatment.py

Outlier and missing-value treatment driven by a per-feature specification.

The PD and LGD feature sets come out of the feature selection, so the cleaning
code must not hard-code them. Each model declares a spec, {feature: [steps]},
and FeatureTreatment applies the steps of the features that are present:

    ("cap_upper_quantile", q)   clip above the q-quantile of the fit data
    ("cap_lower_quantile", q)   clip below the q-quantile of the fit data
    ("cap_upper_value", v)      clip above a fixed economic ceiling
    ("clip_lower_value", v)     clip below a fixed floor (e.g. negatives to 0)
    ("log1p_if_skewed", s)      log1p if the skew after the previous steps exceeds s
    ("replace", {old: new})     recode categories

Every parameter (quantiles, the log1p decision, imputation medians) is computed
in fit() on the reference population and reused unchanged in transform(), so
the same treatment can be applied to any other population (the full portfolio).
Numeric NaNs left after the steps are imputed with the fit median.
"""

import numpy as np
import pandas as pd


class FeatureTreatment:
    """Fit/transform cleaner for a {feature: [steps]} specification."""

    def __init__(self, spec: dict, impute_median: bool = True, verbose: bool = True):
        self.spec = spec
        self.impute_median = impute_median
        self.verbose = verbose
        self.params = {}
        self.fitted = False

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "FeatureTreatment":
        self.params = {}
        for col in df.columns:
            steps = self.spec.get(col, [])
            series = df[col].copy()
            col_params = {}
            for kind, arg in steps:
                if kind == "cap_upper_quantile":
                    col_params["cap_upper"] = float(series.quantile(arg))
                    series = series.clip(upper=col_params["cap_upper"])
                elif kind == "cap_lower_quantile":
                    col_params["cap_lower"] = float(series.quantile(arg))
                    series = series.clip(lower=col_params["cap_lower"])
                elif kind == "cap_upper_value":
                    series = series.clip(upper=arg)
                elif kind == "clip_lower_value":
                    series = series.clip(lower=arg)
                elif kind == "log1p_if_skewed":
                    skew = float(series.skew())
                    col_params["log1p"] = skew > arg
                    if self.verbose:
                        decision = "log1p applied" if col_params["log1p"] else "log1p not applied"
                        print(f"{col}: skew after capping {skew:.4f} -> {decision}")
                    if col_params["log1p"]:
                        series = np.log1p(series)
                elif kind == "replace":
                    series = series.replace(arg)
                else:
                    raise ValueError(f"Unknown treatment '{kind}' for {col}")
            if self.impute_median and pd.api.types.is_numeric_dtype(series) and series.isna().any():
                col_params["impute"] = float(series.median())
            self.params[col] = col_params

        untreated = [c for c in df.columns
                     if c not in self.spec and pd.api.types.is_numeric_dtype(df[c])]
        if self.verbose and untreated:
            print(f"Numeric features without a treatment step: {untreated}")
        self.fitted = True
        return self

    # ── transform ────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FeatureTreatment has not been fitted. Call fit() first.")
        df = df.copy()
        for col in df.columns:
            if col not in self.params:
                continue
            col_params = self.params[col]
            for kind, arg in self.spec.get(col, []):
                if kind == "cap_upper_quantile":
                    df[col] = df[col].clip(upper=col_params["cap_upper"])
                elif kind == "cap_lower_quantile":
                    df[col] = df[col].clip(lower=col_params["cap_lower"])
                elif kind == "cap_upper_value":
                    df[col] = df[col].clip(upper=arg)
                elif kind == "clip_lower_value":
                    df[col] = df[col].clip(lower=arg)
                elif kind == "log1p_if_skewed" and col_params.get("log1p"):
                    df[col] = np.log1p(df[col])
                elif kind == "replace":
                    df[col] = df[col].replace(arg)
            if "impute" in col_params:
                df[col] = df[col].fillna(col_params["impute"])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def summary(self) -> pd.DataFrame:
        """One row per feature: declared steps and fitted parameters."""
        rows = []
        for col, col_params in self.params.items():
            steps = ", ".join(f"{k}({v})" for k, v in self.spec.get(col, [])) or "none"
            rows.append({"feature": col, "steps": steps,
                         **{k: v for k, v in col_params.items()}})
        return pd.DataFrame(rows).set_index("feature")
