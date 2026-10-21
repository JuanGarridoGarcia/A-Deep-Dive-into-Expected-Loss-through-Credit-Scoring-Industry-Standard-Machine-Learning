"""
pipeline_LGD.py

FeaturePipeline_LGD: fit/transform preprocessing for the LGD model.
"""

import pickle

import pandas as pd
from sklearn.preprocessing import RobustScaler

from clean_features_LGD import LGD_TREATMENTS
from feature_treatment import FeatureTreatment

# Numeric columns that are really categories (one-hot encoded, not scaled)
CATEGORICAL_NUMERIC = ["loan_term_months"]


class FeaturePipeline_LGD:
    """
    Feature preprocessing pipeline for the LGD (Loss Given Default) model.

    Fitted on the defaulted loans of the training split only; the fitted
    parameters are then applied unchanged to any population (test split, full
    portfolio). The feature set is taken from the columns passed to fit(), so
    the pipeline follows whatever the feature selection returns.

    Steps:
        1. Outlier and missing-value treatment from LGD_TREATMENTS
           (caps, log1p decision and imputation medians fitted here)
        2. One-hot encoding of categorical features (drop_first)
        3. RobustScaler on the numeric features
    """

    def __init__(self):
        self.params = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline_LGD":
        """Compute every preprocessing parameter from the training population."""
        self.treatment = FeatureTreatment(LGD_TREATMENTS).fit(df)
        df_clean = self.treatment.transform(df)

        categorical = [c for c in df_clean.columns
                       if c in CATEGORICAL_NUMERIC or not pd.api.types.is_numeric_dtype(df_clean[c])]
        numeric = [c for c in df_clean.columns if c not in categorical]

        df_encoded = pd.get_dummies(df_clean, columns=categorical, drop_first=True)
        self.scaler = RobustScaler().fit(df_encoded[numeric])

        self.params = {
            "categorical_features": categorical,
            "numeric_features": numeric,
            "encoded_columns": df_encoded.columns.tolist(),
            "treatment": self.treatment.params,
        }
        self.fitted = True
        return self

    def _apply_treatments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Outlier and missing-value treatment with the fitted parameters."""
        return self.treatment.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted parameters to any dataset."""
        if not self.fitted:
            raise RuntimeError("Pipeline has not been fitted. Call fit() before transform().")

        df = self._apply_treatments(df)
        df = pd.get_dummies(df, columns=self.params["categorical_features"], drop_first=True)
        # Align to the training schema (categories unseen in training become all-zero)
        df = df.reindex(columns=self.params["encoded_columns"], fill_value=0)
        df[self.params["numeric_features"]] = self.scaler.transform(df[self.params["numeric_features"]])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str) -> None:
        if not self.fitted:
            raise RuntimeError("Pipeline has not been fitted. Fit before saving.")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Pipeline saved to {path}")

    @staticmethod
    def load(path: str) -> "FeaturePipeline_LGD":
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        print(f"Pipeline loaded from {path}")
        return pipeline
