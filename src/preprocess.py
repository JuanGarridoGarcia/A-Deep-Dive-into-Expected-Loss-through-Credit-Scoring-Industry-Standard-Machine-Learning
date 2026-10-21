"""
preprocess.py

Scaling and encoding shared by the PD and LGD training notebooks.
"""

import pandas as pd
from sklearn.preprocessing import RobustScaler


def preprocess_features(df_train: pd.DataFrame, df_test: pd.DataFrame,
                        categorical_features: list, numeric_features: list):
    """
    One-hot encode the categorical features and scale the numeric ones with a
    RobustScaler fitted on the training set only.

    Parameters
    ----------
    df_train             : training set (already cleaned)
    df_test              : test set (already cleaned)
    categorical_features : categorical feature names to encode
    numeric_features     : numeric feature names to scale

    Returns
    -------
    df_train_processed, df_test_processed, scaler
    """
    # One-hot encoding, then align the test columns to the training schema
    df_train = pd.get_dummies(df_train, columns=categorical_features, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=categorical_features, drop_first=True)
    df_train, df_test = df_train.align(df_test, join="left", axis=1, fill_value=0)

    # RobustScaler: fit on train only, transform both
    scaler = RobustScaler()
    df_train[numeric_features] = scaler.fit_transform(df_train[numeric_features])
    df_test[numeric_features] = scaler.transform(df_test[numeric_features])

    return df_train, df_test, scaler


# The LGD model uses exactly the same preprocessing on the defaulted population.
preprocess_features_LGD = preprocess_features
