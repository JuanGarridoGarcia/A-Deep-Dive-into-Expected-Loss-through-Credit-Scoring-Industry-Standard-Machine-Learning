import os
import json
import pickle
import numpy as np
import pandas as pd

from scipy.stats import uniform, randint

from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics         import (
    mean_squared_error, mean_absolute_error, r2_score
)

from xgboost  import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


MODELS_DIR = 'models'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stratified_sample_regression(X: pd.DataFrame, y: pd.Series, n: int, random_state: int = 42):
    """
    Returns a random sample of size n from X, y.
    No stratification for regression — samples randomly preserving the target distribution.
    """
    idx = X.sample(n=min(n, len(X)), random_state=random_state).index
    return X.loc[idx], y.loc[idx]


def _evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Computes regression metrics on the test set."""
    y_pred = model.predict(X_test)
    rmse   = mean_squared_error(y_test, y_pred) ** 0.5
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    return {
        'rmse': round(rmse, 4),
        'mae':  round(mae,  4),
        'r2':   round(r2,   4),
    }


def _save(model, name: str, metrics: dict) -> None:
    """Saves fitted model and metrics to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path   = os.path.join(MODELS_DIR, f'{name}.pkl')
    metrics_path = os.path.join(MODELS_DIR, f'{name}_metrics.json')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"  Saved: {model_path} | {metrics_path}")


def _run_search(
    name:         str,
    estimator,
    param_dist:   dict,
    X_train:      pd.DataFrame,
    y_train:      pd.Series,
    X_test:       pd.DataFrame,
    y_test:       pd.Series,
    sample_size:  int,
    n_iter:       int,
    random_state: int = 42,
) -> dict:
    """
    Core training function. Runs RandomizedSearchCV on a random sample,
    refits the best estimator on the full training set, and evaluates on test.
    Optimizes negative RMSE (sklearn convention for minimization problems).
    """
    print(f"\n{'─' * 60}")
    print(f"  Training: {name}")
    print(f"{'─' * 60}")

    # Random sample for hyperparameter search
    print(f"  Sampling {min(sample_size, len(X_train)):,} rows for hyperparameter search...")
    X_sample, y_sample = _stratified_sample_regression(X_train, y_train, n=sample_size, random_state=random_state)

    # RandomizedSearchCV on sample — optimize negative RMSE
    print(f"  Running RandomizedSearchCV ({n_iter} iterations, 5-fold CV)...")
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_root_mean_squared_error',
        cv=5,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_sample, y_sample)

    print(f"  Best params  : {search.best_params_}")
    print(f"  Best CV RMSE : {-search.best_score_:.4f}")

    # Refit best estimator on full training set
    print(f"  Refitting on full training set ({len(X_train):,} rows)...")
    best_model = search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate on test set
    metrics = _evaluate(best_model, X_test, y_test)
    metrics['best_params'] = search.best_params_
    metrics['cv_rmse']     = round(-search.best_score_, 4)

    print(f"  Test RMSE: {metrics['rmse']} | MAE: {metrics['mae']} | R²: {metrics['r2']}")

    _save(best_model, name, metrics)
    return metrics


# ── Individual model trainers ─────────────────────────────────────────────────

def train_ridge(
    X_train, y_train, X_test, y_test,
    sample_size: int = 100_000,
    n_iter: int = 50,
) -> dict:
    """
    Ridge regression used instead of plain linear regression to add L2
    regularization, which improves stability on correlated features.
    """
    param_dist = {
        'alpha': uniform(0.01, 100),
        'fit_intercept': [True, False],
    }

    return _run_search(
        name='ridge_LGD',
        estimator=Ridge(random_state=42),
        param_dist=param_dist,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        sample_size=sample_size,
        n_iter=n_iter,
    )


def train_decision_tree(
    X_train, y_train, X_test, y_test,
    sample_size: int = 100_000,
    n_iter: int = 50,
) -> dict:

    param_dist = {
        'max_depth':         randint(3, 15),
        'min_samples_split': randint(50, 500),
        'min_samples_leaf':  randint(20, 200),
        'criterion':         ['squared_error', 'absolute_error'],
    }

    return _run_search(
        name='decision_tree_LGD',
        estimator=DecisionTreeRegressor(random_state=42),
        param_dist=param_dist,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        sample_size=sample_size,
        n_iter=n_iter,
    )


def train_random_forest(
    X_train, y_train, X_test, y_test,
    sample_size: int = 100_000,
    n_iter: int = 50,
) -> dict:

    param_dist = {
        'n_estimators':      randint(100, 500),
        'max_depth':         randint(5, 20),
        'min_samples_split': randint(50, 500),
        'min_samples_leaf':  randint(20, 200),
        'max_features':      ['sqrt', 'log2', 0.3, 0.5],
    }

    return _run_search(
        name='random_forest_LGD',
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_dist=param_dist,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        sample_size=sample_size,
        n_iter=n_iter,
    )


def train_xgboost(
    X_train, y_train, X_test, y_test,
    sample_size: int = 100_000,
    n_iter: int = 75,
) -> dict:

    param_dist = {
        'n_estimators':     randint(100, 500),
        'max_depth':        randint(3, 10),
        'learning_rate':    uniform(0.01, 0.2),
        'subsample':        uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10),
        'gamma':            uniform(0, 0.5),
        'reg_alpha':        uniform(0, 1),
        'reg_lambda':       uniform(0.5, 2),
    }

    return _run_search(
        name='xgboost_LGD',
        estimator=XGBRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        param_dist=param_dist,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        sample_size=sample_size,
        n_iter=n_iter,
    )


def train_lightgbm(
    X_train, y_train, X_test, y_test,
    sample_size: int = 100_000,
    n_iter: int = 75,
) -> dict:

    param_dist = {
        'n_estimators':      randint(100, 500),
        'max_depth':         randint(3, 10),
        'learning_rate':     uniform(0.01, 0.2),
        'subsample':         uniform(0.6, 0.4),
        'colsample_bytree':  uniform(0.6, 0.4),
        'min_child_samples': randint(20, 200),
        'reg_alpha':         uniform(0, 1),
        'reg_lambda':        uniform(0, 2),
        'num_leaves':        randint(20, 100),
    }

    return _run_search(
        name='lightgbm_LGD',
        estimator=LGBMRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
        param_dist=param_dist,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        sample_size=sample_size,
        n_iter=n_iter,
    )


def train_catboost(
    X_train, y_train, X_test, y_test,
    sample_size: int = 100_000,
    n_iter: int = 75,
) -> dict:

    param_dist = {
        'iterations':    randint(100, 500),
        'depth':         randint(3, 10),
        'learning_rate': uniform(0.01, 0.2),
        'l2_leaf_reg':   uniform(1, 10),
        'subsample':     uniform(0.6, 0.4),
        'border_count':  randint(32, 255),
    }

    return _run_search(
        name='catboost_LGD',
        estimator=CatBoostRegressor(
            random_state=42,
            verbose=0,
            allow_writing_files=False,
        ),
        param_dist=param_dist,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        sample_size=sample_size,
        n_iter=n_iter,
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def train_all_models_LGD(
    X_train:     pd.DataFrame,
    y_train:     pd.Series,
    X_test:      pd.DataFrame,
    y_test:      pd.Series,
    sample_size: int = 100_000,
) -> pd.DataFrame:
    """
    Trains all LGD models sequentially via RandomizedSearchCV on a random sample,
    refits each on the full training set, evaluates on test, and saves models and
    metrics to disk under models/.

    Parameters
    ----------
    X_train, y_train : full training set (cleaned, scaled, encoded)
    X_test,  y_test  : test set
    sample_size      : number of rows for RandomizedSearchCV hyperparameter search

    Returns
    -------
    results_df : DataFrame with one row per model and columns for each metric,
                 sorted by RMSE ascending (lower is better)
    """

    results = {}

    results['ridge']         = train_ridge(X_train, y_train, X_test, y_test, sample_size)
    results['decision_tree'] = train_decision_tree(X_train, y_train, X_test, y_test, sample_size)
    results['random_forest'] = train_random_forest(X_train, y_train, X_test, y_test, sample_size)
    results['xgboost']       = train_xgboost(X_train, y_train, X_test, y_test, sample_size)
    results['lightgbm']      = train_lightgbm(X_train, y_train, X_test, y_test, sample_size)
    results['catboost']      = train_catboost(X_train, y_train, X_test, y_test, sample_size)

    # Build summary DataFrame
    summary_rows = []
    for model_name, metrics in results.items():
        summary_rows.append({
            'model':    model_name,
            'rmse':     metrics['rmse'],
            'mae':      metrics['mae'],
            'r2':       metrics['r2'],
            'cv_rmse':  metrics['cv_rmse'],
        })

    results_df = (
        pd.DataFrame(summary_rows)
        .set_index('model')
        .sort_values('rmse', ascending=True)
    )

    # Save summary
    summary_path = os.path.join(MODELS_DIR, 'LGD_model_comparison.csv')
    results_df.to_csv(summary_path)
    print(f"\n Results saved to {summary_path}")
    print(f"\n{'═' * 60}")
    print(results_df.to_string())
    print(f"{'═' * 60}")

    return results_df
