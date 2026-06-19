# Technical README — Production Notebooks

This document describes the structure, execution order, inputs, outputs, and design decisions of the production notebook pipeline. It is intended as a reference for future development and maintenance of the project.

The production notebooks are a clean, modular reorganization of the original exploratory notebook (`original_notebook/CreditScoring_ExpectedLossCalculation.ipynb`), which contains the full analysis with all intermediate outputs, visualizations, and exploratory comments. The production notebooks are structured for reproducibility and independent execution.

---

## Repository Structure

```
repo/
    original_notebook/
        CreditScoring_ExpectedLossCalculation.ipynb   # full exploratory notebook
    production_notebooks/
        01_data_diagnosis.ipynb
        02_feature_selection.ipynb
        03_eda_pd.ipynb
        04_eda_lgd.ipynb
        05_training_pd.ipynb
        06_training_lgd.ipynb
        07_expected_loss.ipynb
    src/
        pipeline_PD.py              # FeaturePipeline_PD class
        pipeline_LGD.py             # FeaturePipeline_LGD class
        clean_features_PD.py        # outlier treatment function for PD
        clean_features_LGD.py       # outlier treatment function for LGD
        train_models_PD.py          # training orchestration for PD models
        train_models_LGD.py         # training orchestration for LGD models
        preprocess.py               # RobustScaler + One-Hot Encoding functions
    docs/
        pd_feature_descriptions.md
        pd_feature_treatment.md
        lgd_feature_descriptions.md
        lgd_feature_treatment.md
        lgd_feature_selection_justification.md
        outlier_treatment_methodology.md
        inference_strategy.md
        pd_model_results.md
        lgd_model_results.md
        pd_probability_calibration.md
        expected_loss_analysis.md
    data/
        raw/
            target.csv
            X.csv
        processed/                  # generated at runtime — not tracked by git
    models/                         # generated at runtime — not tracked by git
    requirements.txt
    README.md                       # GitHub README
```

---

## Execution Order

The notebooks are designed to be independent — each one loads its required inputs from disk and saves its outputs to disk. They can be opened and executed individually without running the full pipeline from scratch. The recommended execution order is sequential:

```
01 → 02 → 03 → 04 → 05 → 06 → 07
```

Each notebook contains a load checkpoint cell at the bottom that can be run to restore the notebook state without re-executing the full analysis.

---

## Data Flow

```
data/raw/
    target.csv, X.csv
        ↓
01_data_diagnosis
        ↓ checkpoint_end_Data_Diagnosis.pkl
02_feature_selection
        ↓ checkpoint_DataFrame_for_EDA.pkl
        ↓ features_PD_for_EDA.pkl
        ↓ features_LGD_for_EDA.pkl
       ↙                          ↘
03_eda_pd                      04_eda_lgd
       ↓                              ↓
checkpoint_df_PD_training.pkl    checkpoint_df_LGD_training.pkl
       ↓                              ↓
05_training_pd                 06_training_lgd
       ↓                              ↓
checkpoint_df_master_with_PD.pkl      ↓
              ↘                      ↙
        checkpoint_df_master_with_PD_LGD.pkl
                        ↓
                07_expected_loss
                        ↓
               portfolio_with_EL.pkl
```

---

## Notebook Descriptions

### `01_data_diagnosis.ipynb`

**Input:** `data/raw/target.csv`, `data/raw/X.csv`  
**Output:** `data/processed/checkpoint_end_Data_Diagnosis.pkl`

Loads the raw data, constructs the master dataframe with both target variables, and performs the data diagnosis phase. Key operations:

- Construction of `Target_Variable_Default` (binary PD target) and `Target_Variable_Loss` (continuous LGD target, expressed as `100 * max_bal_owed / EAD`)
- Removal of loans with `Target_Variable_Loss >= 300%` (economically implausible)
- Treatment of `late_fees_rec`: binary flag `has_late_fees_PD` for PD, clipped continuous for LGD
- Treatment of `disbursement_method`: binary flag `is_directpay_PD` for PD
- Treatment of `mths_since_*` variables: NaN imputed with sentinel value 999 (MNAR — event never occurred)
- Treatment of `emp_title`: grouped into 12 professional categories via normalized mapping
- Treatment of `emp_length`: converted to numeric, NaN imputed with 0 (unemployment)

The dataset covers Lending Club loans originated between 2007 and 2018 (2,139,643 rows before filtering).

---

### `02_feature_selection.ipynb`

**Input:** `data/processed/checkpoint_end_Data_Diagnosis.pkl`  
**Output:** `data/processed/checkpoint_DataFrame_for_EDA.pkl`, `data/processed/features_PD_for_EDA.pkl`, `data/processed/features_LGD_for_EDA.pkl`

Performs three-stage feature selection for both PD and LGD models:

**Stage 1 — Predictive power analysis:** mutual information, permutation importance, and Lasso coefficients. Conservative AND criterion: a feature must show signal in at least two of the three methods. Boruta run for PD only — LGD omitted due to noisy continuous target.

**Stage 2 — Correlation filter:** unified correlation matrix (Pearson for numeric, Cramér's V for categorical, Eta squared for mixed). Pairs above threshold 0.75 are flagged; the feature with lower mutual information is discarded.

**Stage 3 — Exploratory XGBoost:** SHAP-based feature importance with cumulative threshold cutoff. SHAP is the primary criterion (no cardinality bias). Threshold is 0.80 for PD and 0.90 for LGD — the higher value for LGD reflects the flatter SHAP curve caused by the noisier continuous target and compressed defaulted population.

**Leakage exclusions:** `princ_rec`, `interest_rec`, `remaining_princ_for_tot_amnt_fund` (post-default recovery variables), `has_late_fees_PD` (accumulates after default), `issue_date_year` and `issue_date_month` (survival bias).

**Expert additions:** `dept_paym_income_ratio` added to LGD feature set after confirming low correlation with `annual_income` and `monthly_payment` and its Basel IRB relevance.

**Final feature sets:**

PD model (15 features):
```python
['interest_rate', 'monthly_payment', 'home_ownership_status', 'annual_income',
 'verification_status', 'is_directpay_PD', 'used_credit_share', 'num_inq_in_6mths',
 'num_rev_trades_op_in_12mths', 'num_inq_in_12mths', 'max_bal_owed',
 'dept_paym_income_ratio', 'bal_to_cred_lim', 'emp_length', 'earliest_cr_line_year']
```

LGD model (10 features):
```python
['total_credit_revolving_bal', 'monthly_payment', 'num_open_credit_lines',
 'loan_term_months', 'num_rev_trades_op_in_24mths', 'used_credit_share',
 'interest_rate', 'annual_income', 'emp_length', 'dept_paym_income_ratio']
```

---

### `03_eda_pd.ipynb`

**Input:** `data/processed/checkpoint_DataFrame_for_EDA.pkl`, `data/processed/features_PD_for_EDA.pkl`  
**Output:** `data/processed/checkpoint_df_PD_training.pkl`, `data/processed/features_PD_num_for_training.pkl`, `data/processed/features_PD_cat_for_training.pkl`

EDA and feature engineering for PD model features. Detailed treatment justification is in `docs/pd_feature_treatment.md`. Summary of treatments applied:

| Feature | Treatment |
|---|---|
| `annual_income` | Cap p99, log1p if skew > 2 post-capping |
| `used_credit_share` | Cap at 100 (economic ceiling) |
| `bal_to_cred_lim` | Cap at 100, impute NaNs with median post-capping |
| `num_inq_in_12mths` | Cap at p99 = 11 |
| `max_bal_owed` | Cap at p99 (log1p discarded — distribution pathology) |
| `dept_paym_income_ratio` | Replace negatives with 0 |
| `earliest_cr_line_year` | Cap at p1 (lower tail errors) |
| `home_ownership_status` | Group OTHER into RENT (0.07% frequency) |
| Remaining features | No treatment required |

---

### `04_eda_lgd.ipynb`

**Input:** `data/processed/checkpoint_DataFrame_for_EDA.pkl`, `data/processed/features_LGD_for_EDA.pkl`  
**Output:** `data/processed/checkpoint_df_LGD_training.pkl`, `data/processed/features_LGD_num_for_training.pkl`, `data/processed/features_LGD_cat_for_training.pkl`

EDA and feature engineering for LGD model features. All analysis performed exclusively on the defaulted loan population (~128k observations). Detailed treatment justification is in `docs/lgd_feature_treatment.md`. Summary of treatments applied:

| Feature | Treatment |
|---|---|
| `total_credit_revolving_bal` | Cap at p99 (log1p discarded — same pathology as `max_bal_owed`) |
| `annual_income` | Cap p99, log1p if skew > 2 post-capping |
| `used_credit_share` | Cap at 100 (economic ceiling) |
| `num_open_credit_lines` | Cap at p99 (discontinuous upper tail) |
| `loan_term_months` | Encoding only (binary categorical) |
| Remaining features | No treatment required |

---

### `05_training_pd.ipynb`

**Input:** `data/processed/checkpoint_df_PD_training.pkl`  
**Output:** `models/lightgbm_PD.pkl`, `models/calibrated_model_PD.pkl`, `models/PD_model_comparison.csv`, `data/processed/checkpoint_df_master_with_PD.pkl`

Trains six classification models for PD estimation. Preprocessing: RobustScaler fitted on train only, One-Hot Encoding for categorical features. All models tuned via RandomizedSearchCV (300k stratified sample, 5-fold CV, AUC-ROC as primary metric).

**Results:**

| Model | AUC-ROC | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|---|
| LightGBM | 0.7732 | 0.6619 | 0.1962 | 0.7501 | 0.3110 | 0.2727 |
| XGBoost | 0.7729 | 0.6618 | 0.1960 | 0.7494 | 0.3108 | 0.2722 |
| CatBoost | 0.7712 | 0.6589 | 0.1952 | 0.7533 | 0.3100 | 0.2680 |
| Random Forest | 0.7412 | 0.6921 | 0.1947 | 0.6462 | 0.2992 | 0.2403 |
| Decision Tree | 0.7246 | 0.5980 | 0.1663 | 0.7352 | 0.2712 | 0.2167 |
| Logistic Regression | 0.7036 | 0.6601 | 0.1745 | 0.6274 | 0.2730 | 0.2132 |

**Selected model:** LightGBM (AUC-ROC 0.7732, PR-AUC 0.2727).

**Calibration:** raw `predict_proba` outputs have mean PD of 0.41 vs observed default rate of 0.10, caused by `class_weight='balanced'`. Isotonic regression applied on test set corrects the calibration to mean PD = 0.1017. Known limitation: calibration fitted on test set — see `docs/pd_probability_calibration.md`.

---

### `06_training_lgd.ipynb`

**Input:** `data/processed/checkpoint_df_LGD_training.pkl`, `data/processed/checkpoint_df_master_with_PD.pkl`  
**Output:** `models/catboost_LGD.pkl`, `models/pipeline_LGD.pkl`, `models/LGD_model_comparison.csv`, `data/processed/checkpoint_df_master_with_PD_LGD.pkl`

Trains six regression models for LGD estimation on the defaulted population. Preprocessing: RobustScaler fitted on LGD train only (defaulted population), One-Hot Encoding for `loan_term_months`. All models tuned via RandomizedSearchCV (100k sample, 5-fold CV, RMSE as primary metric).

**Results:**

| Model | RMSE | MAE | R² | CV RMSE |
|---|---|---|---|---|
| CatBoost | 22.9019 | 14.0287 | 0.6651 | 23.0079 |
| LightGBM | 23.1045 | 14.0950 | 0.6592 | 23.1678 |
| XGBoost | 23.1048 | 14.1304 | 0.6591 | 23.1590 |
| Random Forest | 24.2824 | 14.8607 | 0.6235 | 24.2313 |
| Decision Tree | 24.8346 | 15.2095 | 0.6062 | 24.7837 |
| Ridge | 31.5742 | 20.7162 | 0.3635 | 31.3973 |

**Selected model:** CatBoost (RMSE 22.90, R² 0.6651).

**Calibration check:** mean LGD predicted on defaulted population = 45.49 vs mean real = 45.998 (difference < 0.5 percentage points). No calibration required.

**Inference architecture:** the LGD model is applied to the full portfolio (defaulted + non-defaulted) using `FeaturePipeline_LGD` fitted on the defaulted training population. See `docs/inference_strategy.md` for full justification.

---

### `07_expected_loss.ipynb`

**Input:** `data/processed/checkpoint_df_master_with_PD_LGD.pkl`  
**Output:** `data/processed/portfolio_with_EL.pkl`

Computes `EL = PD * (LGD / 100) * EAD` for every loan in the portfolio and validates the results.

**Results:**

| Metric | Value |
|---|---|
| Total EL | 662,377,572 |
| Observed loss (defaulted loans) | 651,402,450 |
| EL / Observed loss ratio | 1.0168 |
| EL rate (portfolio) | 3.52% |

**Segment validation:**

| Loan term | EL rate |
|---|---|
| 36 months | 3.42% |
| 60 months | 3.68% |

| Interest rate quartile | EL rate |
|---|---|
| Q1 (lowest) | 1.38% |
| Q2 | 2.86% |
| Q3 | 4.04% |
| Q4 (highest) | 5.62% |

Both orderings are economically correct: longer-term and higher-rate loans show higher EL rates. The Q1-Q4 gradient (1.38% to 5.62%) confirms that the model discriminates correctly at segment level, not only at portfolio level.

---

## Source Files (`src/`)

| File | Description |
|---|---|
| `pipeline_PD.py` | `FeaturePipeline_PD` class — fit/transform contract for PD preprocessing |
| `pipeline_LGD.py` | `FeaturePipeline_LGD` class — fit/transform contract for LGD preprocessing |
| `clean_features_PD.py` | Outlier treatment function for PD features |
| `clean_features_LGD.py` | Outlier treatment function for LGD features |
| `train_models_PD.py` | Training orchestration for all 6 PD classification models |
| `train_models_LGD.py` | Training orchestration for all 6 LGD regression models |
| `preprocess.py` | `preprocess_features` and `preprocess_features_LGD` — RobustScaler + OHE |

---

## Documentation (`docs/`)

| File | Contents |
|---|---|
| `pd_feature_descriptions.md` | Economic description of each PD model feature |
| `pd_feature_treatment.md` | Treatment decisions for PD features with quantitative justification |
| `lgd_feature_descriptions.md` | Economic description of each LGD model feature |
| `lgd_feature_treatment.md` | Treatment decisions for LGD features with quantitative justification |
| `lgd_feature_selection_justification.md` | Justification of 0.90 threshold and expert feature additions for LGD |
| `outlier_treatment_methodology.md` | General outlier detection and treatment framework |
| `inference_strategy.md` | Dual pipeline architecture and inference workflow |
| `pd_model_results.md` | PD model comparison, metric interpretation, and limitations |
| `lgd_model_results.md` | LGD model comparison, metric interpretation, and limitations |
| `pd_probability_calibration.md` | Isotonic regression calibration — motivation, method, and limitations |
| `expected_loss_analysis.md` | EL framework, validation checks, and portfolio analysis |

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.13.2 required.

---

## Known Limitations

A full list of project limitations is documented in `docs/pd_model_results.md`, `docs/lgd_model_results.md`, and `docs/expected_loss_analysis.md`. The most important ones are:

- **Random train/test split** instead of temporal split — metrics are optimistic relative to a true out-of-time validation
- **LGD proxy target** — `max_bal_owed / EAD` approximates but does not equal real recovery-adjusted LGD
- **Uniform preprocessing** — RobustScaler and One-Hot Encoding applied uniformly across all models as a portfolio simplification
- **Calibration on test set** — isotonic regression calibrator fitted on the test set due to absence of a dedicated calibration split
- **No downturn LGD adjustment** — through-the-cycle estimates, not Basel IRB downturn LGD
- **No probability calibration for LGD** — regression outputs are well-calibrated at the mean but not formally calibrated across the full distribution
