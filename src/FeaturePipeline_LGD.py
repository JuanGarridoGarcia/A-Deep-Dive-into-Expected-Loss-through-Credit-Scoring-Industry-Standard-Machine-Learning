import pandas as pd
from sklearn.preprocessing import RobustScaler

class FeaturePipeline_LGD:
    """
    Feature preprocessing pipeline for the LGD (Loss Given Default) model.
    Fitted on the defaulted loan population only — parameters must not be
    recomputed at inference time regardless of the input population.

    Fixed decisions (no parameters to fit):
        - used_credit_share : cap at 100 (economic ceiling)

    Data-driven decisions (parameters fitted on defaulted training population):
        - total_credit_revolving_bal : cap at p99
        - annual_income              : cap at p99, log1p if skew > 2 post-capping
        - num_open_credit_lines      : cap at p99
    """

    def __init__(self):
        self.params = {}
        self.fitted  = False

    def fit(self, df: pd.DataFrame) -> 'FeaturePipeline_LGD':
        """
        Computes and stores all treatment parameters from the defaulted
        training population. Must be called on training data only.
        """
        # total_credit_revolving_bal — p99 cap, no log1p (distribution pathology)
        self.params['total_credit_revolving_bal_cap'] = df['total_credit_revolving_bal'].quantile(0.99)

        # annual_income — p99 cap, log1p if skew > 2 post-capping
        self.params['annual_income_cap'] = df['annual_income'].quantile(0.99)
        df_temp = df['annual_income'].clip(upper=self.params['annual_income_cap'])
        self.params['annual_income_log'] = df_temp.skew() > 2

        # num_open_credit_lines — p99 cap (discontinuous upper tail)
        self.params['num_open_credit_lines_cap'] = df['num_open_credit_lines'].quantile(0.99)

        # RobustScaler — fitted on cleaned training data
        numeric_features = [
            'total_credit_revolving_bal', 'monthly_payment', 'num_open_credit_lines',
            'num_rev_trades_op_in_24mths', 'used_credit_share', 'interest_rate',
            'annual_income', 'emp_length', 'dept_paym_income_ratio'
        ]
        df_clean = self._apply_treatments(df.copy())
        df_encoded = pd.get_dummies(df_clean, columns=['loan_term_months'], drop_first=True)
        self.params['numeric_features'] = numeric_features
        self.scaler = RobustScaler()
        self.scaler.fit(df_encoded[numeric_features])
        self.params['encoded_columns'] = df_encoded.columns.tolist()

        self.fitted = True
        return self

    def _apply_treatments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all numerical treatments using stored parameters."""
        # total_credit_revolving_bal — cap at p99
        df['total_credit_revolving_bal'] = df['total_credit_revolving_bal'].clip(
            upper=self.params['total_credit_revolving_bal_cap']
        )

        # annual_income — cap at p99, log1p if flagged during fit
        df['annual_income'] = df['annual_income'].clip(upper=self.params['annual_income_cap'])
        if self.params['annual_income_log']:
            df['annual_income'] = np.log1p(df['annual_income'])

        # used_credit_share — economic ceiling at 100
        df['used_credit_share'] = df['used_credit_share'].clip(upper=100)

        # num_open_credit_lines — cap at p99
        df['num_open_credit_lines'] = df['num_open_credit_lines'].clip(
            upper=self.params['num_open_credit_lines_cap']
        )

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies stored parameters to any dataset.
        Raises an error if called before fit.
        """
        if not self.fitted:
            raise RuntimeError("Pipeline has not been fitted. Call fit() before transform().")

        df = df.copy()

        # Apply numerical treatments
        df = self._apply_treatments(df)

        # One-Hot Encoding
        df = pd.get_dummies(df, columns=['loan_term_months'], drop_first=True)

        # Align columns to training schema
        df = df.reindex(columns=self.params['encoded_columns'], fill_value=0)

        # RobustScaler
        df[self.params['numeric_features']] = self.scaler.transform(
            df[self.params['numeric_features']]
        )

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str) -> None:
        if not self.fitted:
            raise RuntimeError("Pipeline has not been fitted. Fit before saving.")
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Pipeline saved to {path}")

    @staticmethod
    def load(path: str) -> 'FeaturePipeline_LGD':
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"Pipeline loaded from {path}")
        return pipeline