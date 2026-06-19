import os
import json
import pickle
import numpy as np
import pandas as pd

from scipy.stats import uniform, randint

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics         import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


MODELS_DIR = 'models'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stratified_sample(X: pd.DataFrame, y: pd.Series, n: int, random_state: int = 42):
    """Returns a stratified sample of size n from X, y."""
    df = X.copy()
    df['__target__'] = y.values

    classes = df['__target__'].unique()
    samples = []
    for c in classes:
        mask  = df['__target__'] == c
        n_c   = int(n * mask.sum() / len(df))
        samples.append(df[mask].sample(n=n_c, random_state=random_state))

    df_sample = pd.concat(samples).sample(frac=1, random_state=random_state).reset_index(drop=True)
    y_sample  = df_sample.pop('__target__')
    return df_sample, y_sample


def _evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Computes classification metrics on the test set."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return {
        'auc':       round(roc_auc_score(y_test, y_prob), 4),
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall':    round(recall_score(y_test, y_pred), 4),
        'f1':        round(f1_score(y_test, y_pred), 4),
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
    name:        str,
    estimator,
    param_dist:  dict,
    X_train:     pd.DataFrame,
    y_train:     pd.Series,
    X_test:      pd.DataFrame,
    y_test:      pd.Series,
    sample_size: int,
    n_iter:      int,
    random_state: int = 42,
) -> dict:
    """
    Core training function. Runs RandomizedSearchCV on a stratified sample,
    then refits the best estimator on the full training set and evaluates on test.
    """
    print(f"\n{'─' * 60}")
    print(f"  Training: {name}")
    print(f"{'─' * 60}")

    # Stratified sample for hyperparameter search
    print(f"  Sampling {sample_size:,} rows for hyperparameter search...")
    X_sample, y_sample = _stratified_sample(X_train, y_train, n=sample_size, random_state=random_state)

    # RandomizedSearchCV on sample
    print(f"  Running RandomizedSearchCV ({n_iter} iterations, 5-fold CV)...")
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=5,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_sample, y_sample)

    print(f"  Best params : {search.best_params_}")
    print(f"  Best CV AUC : {search.best_score_:.4f}")

    # Refit best estimator on full training set
    print(f"  Refitting on full training set ({len(X_train):,} rows)...")
    best_model = search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate on test set
    metrics = _evaluate(best_model, X_test, y_test)
    metrics['best_params']  = search.best_params_
    metrics['cv_auc']       = round(search.best_score_, 4)

    print(f"  Test AUC: {metrics['auc']} | Accuracy: {metrics['accuracy']} | "
          f"Precision: {metrics['precision']} | Recall: {metrics['recall']} | F1: {metrics['f1']}")

    _save(best_model, name, metrics)
    return metrics


# ── Individual model trainers ─────────────────────────────────────────────────

def train_logistic_regression(
    X_train, y_train, X_test, y_test,
    sample_size: int = 200_000,
    n_iter: int = 100,
) -> dict:

    param_dist = {
        'C':            uniform(0.001, 10),
        'penalty':      ['l1', 'l2'],
        'solver':       ['saga'],
        'max_iter':     [500, 1000, 2000],
        'class_weight': ['balanced'],
    }

    return _run_search(
        name='logistic_regression_PD',
        estimator=LogisticRegression(random_state=42),
        param_dist=param_dist,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        sample_size=sample_size,
        n_iter=n_iter,
    )


def train_decision_tree(
    X_train, y_train, X_test, y_test,
    sample_size: int = 200_000,
    n_iter: int = 100,
) -> dict:

    param_dist = {
        'max_depth':        randint(3, 15),
        'min_samples_split': randint(50, 500),
        'min_samples_leaf':  randint(20, 200),
        'class_weight':     ['balanced'],
        'criterion':        ['gini', 'entropy'],
    }

    return _run_search(
        name='decision_tree_PD',
        estimator=DecisionTreeClassifier(random_state=42),
        param_dist=param_dist,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        sample_size=sample_size,
        n_iter=n_iter,
    )


def train_random_forest(
    X_train, y_train, X_test, y_test,
    sample_size: int = 200_000,
    n_iter: int = 100,
) -> dict:

    param_dist = {
        'n_estimators':      randint(100, 500),
        'max_depth':         randint(5, 20),
        'min_samples_split': randint(50, 500),
        'min_samples_leaf':  randint(20, 200),
        'max_features':      ['sqrt', 'log2', 0.3, 0.5],
        'class_weight':      ['balanced'],
    }

    return _run_search(
        name='random_forest_PD',
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_dist=param_dist,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        sample_size=sample_size,
        n_iter=n_iter,
    )


def train_xgboost(
    X_train, y_train, X_test, y_test,
    sample_size: int = 200_000,
    n_iter: int = 100,
) -> dict:

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

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
        name='xgboost_PD',
        estimator=XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='auc',
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
    sample_size: int = 200_000,
    n_iter: int = 100,
) -> dict:

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    param_dist = {
        'n_estimators':     randint(100, 500),
        'max_depth':        randint(3, 10),
        'learning_rate':    uniform(0.01, 0.2),
        'subsample':        uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_samples': randint(20, 200),
        'reg_alpha':        uniform(0, 1),
        'reg_lambda':       uniform(0, 2),
        'num_leaves':       randint(20, 100),
    }

    return _run_search(
        name='lightgbm_PD',
        estimator=LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
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
    sample_size: int = 200_000,
    n_iter: int = 100,
) -> dict:

    param_dist = {
        'iterations':   randint(100, 500),
        'depth':        randint(3, 10),
        'learning_rate': uniform(0.01, 0.2),
        'l2_leaf_reg':  uniform(1, 10),
        'subsample':    uniform(0.6, 0.4),
        'border_count': randint(32, 255),
    }

    return _run_search(
        name='catboost_PD',
        estimator=CatBoostClassifier(
            random_state=42,
            verbose=0,
            auto_class_weights='Balanced',
        ),
        param_dist=param_dist,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        sample_size=sample_size,
        n_iter=n_iter,
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def train_all_models_PD(
    X_train:     pd.DataFrame,
    y_train:     pd.Series,
    X_test:      pd.DataFrame,
    y_test:      pd.Series,
    sample_size: int = 200_000,
) -> pd.DataFrame:
    """
    Trains all PD models sequentially via RandomizedSearchCV on a stratified sample,
    refits each on the full training set, evaluates on test, and saves models and
    metrics to disk under models/.

    Parameters
    ----------
    X_train, y_train : full training set (cleaned, scaled, encoded)
    X_test,  y_test  : test set
    sample_size      : number of rows for RandomizedSearchCV hyperparameter search

    Returns
    -------
    results_df : DataFrame with one row per model and columns for each metric
    """

    results = {}

    results['logistic_regression'] = train_logistic_regression(X_train, y_train, X_test, y_test, sample_size)
    results['decision_tree']       = train_decision_tree(X_train, y_train, X_test, y_test, sample_size)
    results['random_forest']       = train_random_forest(X_train, y_train, X_test, y_test, sample_size)
    results['xgboost']             = train_xgboost(X_train, y_train, X_test, y_test, sample_size)
    results['lightgbm']            = train_lightgbm(X_train, y_train, X_test, y_test, sample_size)
    results['catboost']            = train_catboost(X_train, y_train, X_test, y_test, sample_size)

    # Build summary DataFrame
    summary_rows = []
    for model_name, metrics in results.items():
        summary_rows.append({
            'model':      model_name,
            'auc':        metrics['auc'],
            'accuracy':   metrics['accuracy'],
            'precision':  metrics['precision'],
            'recall':     metrics['recall'],
            'f1':         metrics['f1'],
            'cv_auc':     metrics['cv_auc'],
        })

    results_df = (
        pd.DataFrame(summary_rows)
        .set_index('model')
        .sort_values('auc', ascending=False)
    )

    # Save summary
    summary_path = os.path.join(MODELS_DIR, 'PD_model_comparison.csv')
    results_df.to_csv(summary_path)
    print(f"\n Results saved to {summary_path}")
    print(f"\n{'═' * 60}")
    print(results_df.to_string())
    print(f"{'═' * 60}")

    return results_df
