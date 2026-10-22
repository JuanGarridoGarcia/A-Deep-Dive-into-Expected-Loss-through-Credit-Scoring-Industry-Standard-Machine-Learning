# Technical README: Production Notebooks

This document describes the execution order, inputs, outputs and design decisions of the production notebooks. It is the reference for anyone who wants to rerun, audit or extend the pipeline. The figures quoted here come from the run committed with the notebooks (Python 3.13, `requirements.txt`).

The production notebooks are a modular rewrite of the original exploratory notebook (`original_notebook/CreditScoring_ExpectedLossCalculation.ipynb`), which is kept as a record. The two differ in method, not only in layout: the production version uses a net-loss LGD target, develops both models on matured loans only and calibrates the PD on a dedicated split (see "What changed from the original notebook" at the end).

---

## Repository Structure

```
repo/
    original_notebook/
        CreditScoring_ExpectedLossCalculation.ipynb   # archived exploratory run
    production_notebooks/
        01_data_diagnosis.ipynb
        02_feature_selection.ipynb
        03_eda_pd.ipynb
        04_eda_lgd.ipynb
        05_training_pd.ipynb
        06_training_lgd.ipynb
        07_expected_loss.ipynb
        TECHNICAL_README.md
    src/
        feature_engineering.py      # employment title grouping, emp_length parsing, flags
        feature_statistics.py       # dataset audit, outlier summary, distribution tests
        feature_plot.py             # EDA plots
        feature_selection.py        # MI, permutation, Lasso, Boruta, correlation filter, SHAP cutoff
        feature_treatment.py        # FeatureTreatment: spec-driven caps, log1p and imputation (fit/transform)
        clean_features_PD.py        # PD_TREATMENTS specification
        clean_features_LGD.py       # LGD_TREATMENTS specification
        preprocess.py               # one-hot encoding + RobustScaler (fit on train, apply anywhere)
        pipeline_LGD.py             # FeaturePipeline_LGD: treatment, encoding and scaling for LGD
        train_models_PD.py          # tuning and training of the 6 PD classifiers
        train_models_LGD.py         # tuning and training of the 6 LGD regressors
        explain.py                  # SHAP values for linear or tree models, feature-level importance
        lgd_analysis.py             # decomposition of the LGD into default timing and recovery
        readme_figures.py           # every chart in the README, drawn from the notebook outputs
    tests/                          # unit tests for src/ (pytest)
    assets/                         # README charts (written by the notebooks)
    data/
        raw/                        # X.csv, target.csv (not tracked)
        processed/                  # checkpoints written at runtime (not tracked)
    models/                         # fitted models and comparison tables (not tracked)
    requirements.txt
    requirements-dev.txt
    README.md
```

---

## Execution Order and Data Flow

The notebooks run in order, 01 to 07. Each one reads its inputs from `data/processed/` and writes its outputs there, so any notebook can be rerun on its own once its inputs exist. Each ends with a cell that reloads its outputs without recomputing them.

```
data/raw/  X.csv, target.csv
        |
01_data_diagnosis ............. checkpoint_end_Data_Diagnosis.pkl
        |
02_feature_selection .......... checkpoint_DataFrame_for_EDA.pkl
        |                       features_PD_for_EDA.pkl, features_LGD_for_EDA.pkl
        +-------------------------------+
        |                               |
03_eda_pd                          04_eda_lgd
  checkpoint_df_PD_training.pkl      checkpoint_df_LGD_training.pkl
  features_PD_{num,cat}_...pkl       features_LGD_{num,cat}_...pkl
        |                               |
05_training_pd                          |
  checkpoint_df_master_with_PD.pkl -----+
                                        |
                                 06_training_lgd
                                   checkpoint_df_master_with_PD_LGD.pkl
                                        |
                                 07_expected_loss
                                   portfolio_with_EL.pkl
```

Runtime on a 16-core desktop: about three hours: 02 takes 30 minutes (Boruta), 05 about 115 and 06 about 30, dominated by the PD hyperparameter search in 05.

---

## Two Design Decisions That Run Through Every Notebook

### Development on matured loans only

The data ends with loans issued in December 2018, and that last issue month is used as the snapshot date. A loan issued in 2017 with a 60-month term has most of its life ahead of it: if it is labelled "not defaulted", that only means "not defaulted yet". Training on such loans teaches a model that recent loans are safe, and the LGD of recent defaults is truncated because their recoveries are still coming in.

`01_data_diagnosis` therefore flags a loan as **matured** when `issue date + term <= snapshot`. In practice that means 36-month loans issued up to 2015 and 60-month loans issued up to 2013: 676,113 loans (31.6% of the portfolio), with a default rate of 14.9% and 100,589 defaults. Feature selection, outlier treatment and both models are developed on this sample. The fitted models then score **every** loan, matured or not, which is how a lender would value the portfolio.

### No information from after origination

Every feature must be known when the loan is granted. Payment and recovery variables (`princ_rec`, `interest_rec`, `paym_rec_for_tot_amnt_fund`, `remaining_princ_for_tot_amnt_fund`, late fees) define the targets and are never features. Ten bureau fields that Lending Club only started reporting in 2016 (`max_bal_owed`, `bal_to_cred_lim`, `num_inq`, the recent-trade counts and the months since the last instalment account) are missing for almost every matured loan; they are excluded because their missingness would encode the vintage, not the borrower. The issue date is excluded for the same reason.

---

## Notebook Descriptions

### `01_data_diagnosis.ipynb`

**Input:** `data/raw/target.csv`, `data/raw/X.csv`
**Output:** `data/processed/checkpoint_end_Data_Diagnosis.pkl`

- **PD target:** `Target_Variable_Default`, binary.
- **EAD:** `funded_amnt`, the amount at origination.
- **LGD target:** net loss on a defaulted loan as a share of EAD. Recoveries are what remains of total payments after principal, interest and late fees:
  ```
  recoveries = paym_rec_for_tot_amnt_fund - princ_rec - interest_rec - late_fees_rec
  net_loss   = EAD - princ_rec - recoveries        (clipped to [0, EAD])
  LGD (%)    = 100 * net_loss / EAD
  ```
  A sanity check asserts that implied recoveries are zero for non-defaulted loans. Collection fees are not in the data, so recoveries are gross and the LGD is slightly understated.
- **Matured flag** and the evidence for it: default rate and LGD by issue year.
- **Feature engineering:** `late_fees_rec` as a binary flag for PD and a clipped amount for LGD; `disbursement_method` as a binary flag; `mths_since_*` NaNs set to 999 (MNAR: the event never happened); `emp_title` grouped into professional categories; `emp_length` parsed to years.

### `02_feature_selection.ipynb`

**Input:** `checkpoint_end_Data_Diagnosis.pkl`
**Output:** `checkpoint_DataFrame_for_EDA.pkl`, `features_PD_for_EDA.pkl`, `features_LGD_for_EDA.pkl`

Runs on the matured loans (PD) and on the matured, defaulted loans (LGD).

1. **Exclusions:** post-origination variables, the ten vintage-dependent fields, the issue date. A table shows the share of missing values of the vintage-dependent fields by issue year.
2. **Predictive power:** mutual information, permutation importance (random forest, all cores) and Lasso; Boruta for PD. A feature is kept if it shows signal in at least two of the three methods.
3. **Correlation filter:** a single matrix mixing Pearson (numeric), Cramér's V (categorical) and the correlation ratio (mixed). In each pair above 0.75 the feature with the lower mutual information is dropped.
4. **Exploratory gradient boosting with a SHAP cutoff:** features are kept up to 80% of cumulative mean |SHAP| for PD and 90% for LGD (the LGD target is noisier, so its importance curve is flatter).
5. **Expert additions:** `dept_paym_income_ratio` for LGD and `loan_term_months` for PD when the cutoff leaves them out. The term sits just below the PD cutoff (it is inside the gain cutoff), and a run without it underestimated the PD of 60-month loans by 4 to 6 points in every matured vintage.

**Final feature sets:**

PD (11 features: 10 from the SHAP cutoff plus the term):
```python
['grade', 'annual_income', 'home_ownership_status', 'num_open_credit_lines',
 'dept_paym_income_ratio', 'monthly_payment', 'num_inq_in_12mths', 'emp_title',
 'total_credit_revolving_bal', 'used_credit_share', 'loan_term_months']
```

LGD (16 features):
```python
['interest_rate', 'loan_term_months', 'num_open_credit_lines', 'total_credit_revolving_bal',
 'num_inq_in_12mths', 'tot_num_credit_lines', 'used_credit_share', 'annual_income',
 'emp_length', 'earliest_cr_line_year', 'mths_since_last_delinq_LGD', 'loan_purpose',
 'addr_state', 'monthly_payment', 'mths_since_recent_revol_delinq_LGD', 'dept_paym_income_ratio']
```

The correlation filter removes `interest_rate` from PD (0.95 with `grade`) and `grade` from LGD; each model keeps the one with the higher mutual information for its own target.

### `03_eda_pd.ipynb` and `04_eda_lgd.ipynb`

**Inputs:** `checkpoint_DataFrame_for_EDA.pkl` and the feature list of each model
**Outputs:** the treated training frames and the numeric / categorical feature lists

The treatment follows a specification per feature (`PD_TREATMENTS`, `LGD_TREATMENTS`) instead of code written for one feature list, so a different selection never breaks the cleaning. `FeatureTreatment` fits every parameter (quantile caps, the log1p decision, imputation medians) on the reference population and reuses it unchanged on any other population:

| Step | Features | Reason |
|---|---|---|
| Cap at p99 | balances and counts with long upper tails | discontinuous tails; log1p rejected for balances because the mass near zero inverts the skew |
| Cap at p99, then log1p if skew > 2 | `annual_income` | heavy right tail; after capping the skew is below 2, so log1p is not applied |
| Cap at 100 | `used_credit_share`, `bal_to_cred_lim` | utilisation above 100% is an economic ceiling |
| Clip negatives to 0 | `dept_paym_income_ratio` | negative ratios are data errors |
| Floor at p1 | `earliest_cr_line_year` | lower-tail data errors |
| OTHER → RENT | `home_ownership_status` | 0.07% of loans, closest profile |
| One-hot encoding | `grade`, `emp_title`, `home_ownership_status`, `loan_term_months`, `loan_purpose`, `addr_state` | categorical (the term is stored as a number but is a category) |

The PD treatment is fitted on the matured loans and applied to every loan. Each notebook prints the tail statistics before and after treatment.

### `05_training_pd.ipynb`

**Input:** `checkpoint_df_PD_training.pkl` and the PD feature lists
**Output:** `models/<model>_PD.pkl`, `models/calibrated_model_PD.pkl`, `models/PD_model_comparison.csv`, `checkpoint_df_master_with_PD.pkl`

- **Splits** of the matured loans, stratified: train 70%, calibration 10%, test 20%.
- **Preprocessing:** one-hot encoding and a RobustScaler fitted on the training split only.
- **Six classifiers** (logistic regression, decision tree, random forest, XGBoost, LightGBM, CatBoost), each tuned with `RandomizedSearchCV` on a 300,000-row stratified sample, 5-fold CV, AUC-ROC. 100 iterations per model except the random forest (20): its search dominated the runtime and it ranks below the boosting models in every run.
- **Selection** by cross-validated AUC. The test split plays no role in the choice.

| Model | CV AUC | Test AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **Logistic regression** (selected) | **0.6849** | 0.6849 | 0.2637 | 0.2254 | 0.6681 | 0.3371 |
| CatBoost | 0.6843 | 0.6859 | 0.2671 | 0.2303 | 0.6563 | 0.3410 |
| XGBoost | 0.6840 | 0.6857 | 0.2668 | 0.2304 | 0.6499 | 0.3402 |
| LightGBM | 0.6839 | 0.6855 | 0.2677 | 0.2299 | 0.6551 | 0.3404 |
| Random forest (20 iterations) | 0.6769 | 0.6770 | 0.2566 | 0.2250 | 0.6564 | 0.3351 |
| Decision tree | 0.6426 | 0.6483 | 0.2305 | 0.2076 | 0.6501 | 0.3147 |

Threshold metrics use 0.5 on the class-weighted scores, so precision and recall describe the raw classifier, not the calibrated PD. The top four are within 0.001 of AUC, below the sampling error of the test AUC (about 0.003): a tie, resolved in favour of the most auditable model. The notebook explains why boosting adds nothing here (`grade` is already a risk score; XGBoost and CatBoost settle on depth-3 trees).

- **Calibration:** the models are trained with class weighting, so their raw probabilities are inflated. An isotonic calibrator is fitted on the calibration split with the trained model wrapped in `FrozenEstimator` (no refit), and evaluated on the untouched test split.

| Test split | Raw | Calibrated |
|---|---|---|
| Mean PD (default rate 0.1488) | 0.4629 | 0.1485 |
| Brier score (constant PD: 0.1267) | 0.2257 | 0.1197 |
| AUC | 0.6849 | 0.6846 |

Isotonic calibration maps some scores to exactly 0 and 1. Regulatory IRB models floor the PD (0.05% under the final Basel III reform, 0.03% under Basel II); the floor is not applied here.

- **SHAP:** the explainer follows the selected model (`LinearExplainer` for a linear model, values in log-odds; `TreeExplainer` for a tree ensemble). The same 50,000 test loans are also explained with the best model of the other family, and the two feature-level rankings (one-hot columns summed back into their feature) are compared.

Share of mean |SHAP| by feature (one-hot columns summed), 50,000 test loans:

| Feature | Logistic regression | CatBoost |
|---|---|---|
| grade | 37.4% | 23.8% |
| home_ownership_status | 10.1% | 8.5% |
| emp_title | 8.2% | 6.5% |
| annual_income | 7.9% | 11.1% |
| dept_paym_income_ratio | 7.3% | 6.7% |
| monthly_payment | 5.8% | 7.5% |
| num_open_credit_lines | 5.8% | 7.3% |
| num_inq_in_12mths | 5.5% | 7.3% |
| total_credit_revolving_bal | 4.7% | 7.6% |
| loan_term_months | 4.4% | 5.8% |
| used_credit_share | 2.6% | 8.0% |

`grade` leads in both; below it the importances are flat and their order differs (Spearman 0.38).

- **Scoring:** every loan in the portfolio gets a calibrated PD. `split_PD` records whether a loan was in train, calibration, test or not matured, so 07 can validate on the test loans alone.

### `06_training_lgd.ipynb`

**Input:** `checkpoint_df_LGD_training.pkl`, `checkpoint_df_master_with_PD.pkl` and the LGD feature lists
**Output:** `models/<model>_LGD.pkl`, `models/pipeline_LGD.pkl`, `models/LGD_model_comparison.csv`, `checkpoint_df_master_with_PD_LGD.pkl`

- **Split** of the matured, defaulted loans: train 80%, test 20%.
- **`FeaturePipeline_LGD`** (treatment, one-hot encoding, RobustScaler) is fitted on the raw training rows and transforms the test split and the whole portfolio, so training and scoring cannot drift apart.
- **Six regressors** (ridge, decision tree, random forest, XGBoost, LightGBM, CatBoost) tuned with `RandomizedSearchCV`, 5-fold CV, RMSE; selection by CV RMSE. The decision-tree search uses squared error only: the absolute-error criterion takes hours per fit on this sample and does not match the RMSE objective.

| Model | CV RMSE | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|
| **CatBoost** (selected) | **22.69** | 22.64 | 18.89 | 0.064 |
| XGBoost | 22.70 | 22.66 | 18.89 | 0.063 |
| LightGBM | 22.71 | 22.65 | 18.89 | 0.063 |
| Random forest | 22.74 | 22.68 | 18.93 | 0.061 |
| Ridge | 22.75 | 22.70 | 18.95 | 0.059 |
| Decision tree | 22.87 | 22.83 | 19.04 | 0.049 |
| Predict the training mean | | 23.41 | | 0 |

- **Why the R² is low** (section 6 of the notebook). The LGD is measured against the amount funded, so `LGD = (1 - a)(1 - r)`, with `a` the share of principal repaid before default and `r` the recovery on the balance at default.

| | Mean | Variance of the LGD explained alone | Test R² from the LGD features |
|---|---|---|---|
| LGD (% of EAD) | 52.2 | | 0.064 |
| `a`, share repaid before default | 0.40 | 0.86 | 0.076 |
| instalments paid before default | 17.5 | 0.75 | 0.084 |
| `r`, recovery on the balance at default | 0.13 | 0.30 | |
| loss on the balance at default, `100(1 - r)` | 87.4 | | 0.006 |

The LGD falls from 82% for defaults in the first six months to 24% for defaults after two to three years. The timing is barely predictable from the application, and the recovery is nearly constant (86% of defaults recover less than 20% of the balance). The low R² is a property of the target, not of the models. The notebook describes the Basel structure (EAD at default from the amortisation schedule, LGD = `1 - r`) as the improvement not implemented.

Calibration of the level: mean predicted LGD 52.2 against 52.2 observed, on the development sample and on the test split. The medians differ (51.7 against 54.3): with an R² of 0.06 the predictions are compressed towards the mean.

- **Inference:** the model scores the whole portfolio through the fitted pipeline (standard IRB practice: the loss if each loan defaulted). Predictions are clipped to [0, 100]. The mean predicted LGD is checked against the observed mean on the development sample and on the test split.

### `07_expected_loss.ipynb`

**Input:** `checkpoint_df_master_with_PD_LGD.pkl`
**Output:** `portfolio_with_EL.pkl` and the EL charts

`EL = PD × LGD / 100 × EAD`, a lifetime figure per loan.

**Portfolio:**

| Metric | Value |
|---|---|
| Total EAD | 31,628,836,375 |
| Total EL | 3,221,170,690 |
| EL rate | 10.18% (matured 7.89%, not matured 11.07%) |
| Share of matured EL carried by loans that did not default | 78.7% |

**Against realised losses** (matured loans, whose outcome is complete):

| Population | Expected loss | Realised loss | Ratio |
|---|---|---|---|
| Matured loans | 696,244,482 | 700,895,593 | 0.993 |
| PD test split | 138,943,183 | 140,684,130 | 0.988 |

**Segments:**

| Segment | EL rate |
|---|---|
| 36 months | 7.13% |
| 60 months | 15.01% |
| Interest-rate Q1 / Q2 / Q3 / Q4 | 3.07% / 7.15% / 11.18% / 18.20% |

The mean PD matches the observed default rate in every interest-rate quartile of the matured loans (Q1 0.062 against 0.065, Q4 0.261 against 0.260).

**Calibration by term and vintage (section 4.4).** On matured loans the PD is within 2 points of the observed default rate in every vintage from 2009 on, for both terms; the test split is calibrated by term (36 months 0.139 against 0.139, 60 months 0.252 against 0.253). For loans still running the observed rate is a lower bound; the 60-month loans of 2014-2015 and the 36-month loans of 2016 have already reached their predicted PD with time left, so those vintages will end somewhat above it.

**Matured against running loans (section 4.5).** The running loans have a higher EL rate mainly through composition (37% are 60-month loans against 8%, weaker grades); within the same term and grade the gap is smaller and positive in 13 of 14 cells, because their application features are weaker on average.

The notebook also writes `pipeline.png`, `key_metrics.png` and `el_by_quartile.png` for the README.

---

## Source Files (`src/`)

| File | Contents |
|---|---|
| `feature_engineering.py` | `group_emp_title`, `clean_emp_length` and the flags built in 01 |
| `feature_statistics.py` | `dataset_audit`, `outlier_summary`, `compare_distributions` |
| `feature_plot.py` | EDA plots |
| `feature_selection.py` | `select_features` (MI, permutation, Lasso, Boruta), correlation matrix and pair analysis, `exploratory_feature_selection` (gain and SHAP cutoffs) |
| `feature_treatment.py` | `FeatureTreatment`: fit/transform of a per-feature treatment specification |
| `clean_features_PD.py`, `clean_features_LGD.py` | `PD_TREATMENTS`, `LGD_TREATMENTS` and their fit helpers |
| `preprocess.py` | `preprocess_features` (fit) and `apply_preprocessing` (apply to any frame) |
| `pipeline_LGD.py` | `FeaturePipeline_LGD` |
| `train_models_PD.py`, `train_models_LGD.py` | search, refit, test metrics and saving of every model; comparison table ranked by CV score |
| `explain.py` | `shap_values_for` (explainer chosen from the model), `feature_level_shap` |
| `lgd_analysis.py` | `lgd_components`, `variance_explained`, `lgd_by_months_paid` |
| `readme_figures.py` | the README charts, including the pipeline diagram |

The unit tests (`pytest`, 31 tests) cover the treatment specification and its fit/transform contract, the LGD pipeline, the preprocessing alignment, the training helpers, the SHAP helpers and the LGD decomposition.

---

## Known Limitations

- **No out-of-time validation.** Splits are random within the matured loans. Later vintages performed worse with the same application profile (section 4.4 of 07), so the EL of the running loans is more likely understated. A vintage adjustment from early-default survival curves is the natural next step.
- **LGD structure.** LGD against the amount funded mixes default timing and recovery; the Basel structure (EAD at default, LGD on that balance) is the proposed improvement (section 6 of 06).
- **EAD = amount funded**, without amortisation.
- **No PD floor.** Isotonic calibration yields PDs of exactly 0; IRB models floor them at 0.05% (0.03% under Basel II).
- **No downturn LGD**: through-the-cycle estimates.
- **Approved loans only**: no reject inference.
- **Application data only**: no behavioural variables.
- **Gross recoveries**: collection fees are not in the data.
- **LGD scored on performing loans** through a pipeline fitted on defaults (standard IRB practice); performing loans can sit outside the feature ranges seen on defaults, which the fitted caps only partly control.
- **Uniform preprocessing** (one-hot encoding and RobustScaler for every model), a simplification that tree models do not need.

---

## What Changed from the Original Notebook

| | Original notebook | Production notebooks |
|---|---|---|
| LGD target | `max_bal_owed / funded_amnt`, partly an accounting identity of the features (R² 0.665) | net loss after principal and recoveries (R² 0.064, explained in 06) |
| Development sample | all loans, including recent vintages whose outcome is incomplete | matured loans only; every loan is scored |
| Vintage-dependent fields | used as features | excluded (their missingness encodes the vintage) |
| PD calibration | fitted on the test set | fitted on a dedicated split, evaluated on the test set |
| Model selection | by test metrics | by cross-validated metrics |
| Outlier treatment | written per feature | specification fitted on the development sample, applied to any population |
| README charts | edited by hand | drawn by the notebooks |
| Headline numbers | AUC 0.773, LGD R² 0.665, EL rate 3.52% | AUC 0.685, LGD R² 0.064, EL rate 10.18% |

The headline numbers fell because the original ones were inflated by the issues listed in the left column, not because the models got worse.
