# Explainable Machine Learning fOr Suspicious Transaction Risk Detection in Anti-Money Laundering

**Umm E Kulsoom** (B01040340)
MSc Computer Science and Technology — Ulster University, Birmingham Campus
Supervisor: Dr Anwar Haq

A reproducible, imbalance-aware and explainable machine-learning pipeline for
suspicious-transaction risk detection on the AMLNet synthetic dataset. All model
outputs are framed as decision-support evidence for human AML reviewers, not as
automated compliance or legal determinations.

> **Reviewing this project?** A [recorded dashboard walkthrough](https://drive.google.com/file/d/10jZbtqbLWENUH5QFAOESdcQHTcB09NLs/view)
> shows the complete tool in use. All reported results are committed under
> `outputs/`, so they can be checked without installing anything.

## Headline result

Class-weighted XGBoost, selected automatically on validation PR-AUC at an
F1-optimal threshold of 0.99, evaluated once on the untouched test set:

| Metric | Value |
|---|---|
| Precision | 0.971 |
| Recall | 0.897 |
| F1-score | 0.933 |
| PR-AUC | 0.952 |
| ROC-AUC | 0.999 |

242 alerts raised from 163,526 test transactions (14.8 per 10,000), of which
97.1% were genuine laundering cases: 235 true positives, 7 false positives,
27 false negatives.

## Getting the dataset

**The dataset is not included in this repository.** The raw file is about
691 MB, which is over GitHub's 100 MB per-file limit, and the AMLNet licence
(CC BY-NC 4.0) is best respected by pointing to the original source rather than
redistributing a copy.

Download it from Zenodo:

- **AMLNet: Synthetic anti-money laundering transaction dataset**
- Author: Huda, S. (2025)
- DOI: https://doi.org/10.5281/zenodo.16736515
- Licence: CC BY-NC 4.0 (non-commercial, attribution required)
- File: `AMLNet.csv` — 691.3 MB (659.3 MiB)
- MD5: `7668fc7d74c787e07546ce85c6f790b9`

Place the downloaded file at:

```
data/raw/AMLNet.csv
```

`src/step00_setup.py` checks the file against the MD5 checksum above before any
processing begins, so a wrong or corrupted download is caught immediately.

### Dataset summary

| Property | Value |
|---|---|
| Transactions (published) | 1,090,173 |
| Transactions (after cleaning) | 1,090,172 — one record with a missing target removed |
| Laundering-positive | 1,745 (0.16%) |
| Normal | 1,088,427 |
| Imbalance ratio | 623.74 : 1 |
| Columns | 17 original |
| Payment types | 8 |
| Transaction categories | 11 |
| Simulation span | 195 days |

## Running the project — step by step

### Step 1 — Prerequisites

- **Python 3.14.5** (the version the results were produced with)
- About **2 GB of free disk space** — 691 MB for the dataset, the rest for
  parquet intermediates and model files
- **16 GB of RAM recommended.** No GPU is needed; every result in this project
  was produced on a standard Windows 11 laptop CPU

Check your Python version:

```bash
python --version
```

### Step 2 — Get the code

```bash
git clone https://github.com/Kulsoom-UEB/AMLNet-Umm-E-Kulsoom-Dissertation.git
cd AMLNet-Umm-E-Kulsoom-Dissertation
```

### Step 3 — Create an isolated environment

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

You should see `(.venv)` at the start of your prompt.

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This takes a few minutes. Versions are pinned so the environment matches the
one used to produce the reported results.

### Step 5 — Download the dataset

1. Go to https://doi.org/10.5281/zenodo.16736515
2. Download `AMLNet.csv`
3. Create the folder and place the file so the path is exactly:

```
data/raw/AMLNet.csv
```

```bash
mkdir -p data/raw data/processed models
```

### Step 6 — Verify the setup

```bash
python src/step00_setup.py
```

This checks the Python and package versions, creates any missing folders, and
verifies `AMLNet.csv` against its published MD5 checksum. **If this step fails,
stop and fix it before continuing** — a wrong or partial download will produce
misleading results downstream.

### Step 7 — Run the pipeline

Everything in one command:

```bash
python src/run_all.py
```

The script prints a banner and a timing line for each stage, then a total
runtime and the output locations.

Useful flags:

```bash
python src/run_all.py --from 05    # resume from step 05 onwards
python src/run_all.py --only 08    # re-run just one step
```

Any stage can also be run directly:

```bash
python src/step04_split.py
```

### What each step does and produces

| Step | Stage | Writes |
|---|---|---|
| 00 | Setup and reproducibility | Environment table, dataset integrity check |
| 01 | Data understanding and leakage audit | Profiling tables, `fig01`–`fig03` |
| 02 | Leakage removal and cleaning | `data/processed/cleaned_amlnet.parquet` |
| 03 | Feature engineering | `data/processed/engineered_amlnet.parquet` |
| 04 | Train / validation / test split | Six split files, split integrity table |
| 05 | Model development and tuning | Six tuned pipelines, hyperparameter results |
| 06 | Threshold optimisation and selection | `models/final_model_config.json`, threshold tables, `fig04`–`fig06` |
| 07 | Final test evaluation | Test metrics, confusion matrix, `fig07`–`fig08` |
| 08 | Explainability | Importance tables, `fig09`–`fig12` |

> **Why the search budget is small.** Step 05 samples six hyperparameter
> configurations per model and strategy, and tunes on a 250,000-row stratified
> subsample rather than the full training set. This is a deliberate hardware
> choice, not an oversight: a larger initial search fitted on all 763,120
> training rows ran out of the 16 GB of available memory and stopped the
> environment mid-run, because the one-hot expanded matrix has to be held
> across three cross-validation folds and several candidate fits at once.
> Reducing the budget and subsampling for the search, then refitting the
> winning configuration on the complete training set, made the six-way
> comparison finish reliably while still producing cross-validated PR-AUC
> between 0.925 and 0.959 across all candidates. If you are running this on
> better-provisioned hardware, `N_ITER` in `src/step05_model_development.py`
> can safely be raised.

The pipeline is deterministic: seed 42 is fixed across `random`, `numpy`,
`PYTHONHASHSEED`, scikit-learn and XGBoost, so a re-run reproduces the same
numbers.

### Step 8 — Launch the dashboard

```bash
streamlit run dashboard/app.py
```

This opens the dashboard in your browser at `http://localhost:8501`.
There is also a copy of the dashboard in `src/app.py`, so
`streamlit run src/app.py` works the same way. See the dashboard section below
for what each tab shows.

### Step 9 — Check the results against the report

After the pipeline finishes, these files should reproduce the headline figures:

| To verify | Look at |
|---|---|
| Final test metrics | `outputs/tables/final_test_metrics.csv` |
| Confusion matrix (235 / 7 / 27) | `outputs/figures/fig07_test_confusion_matrix.png` |
| Model selection by PR-AUC | `outputs/tables/validation_*_optimised*.csv` |
| Selected model and threshold | `models/final_model_config.json` |
| Cross-method importance | `outputs/tables/explainability_*.csv` |

`outputs/` is committed to this repository, so the reported results can be
inspected without running anything — running the pipeline regenerates them.

### Troubleshooting

| Problem | Cause and fix |
|---|---|
| `FileNotFoundError` on `AMLNet.csv` | The dataset is not at `data/raw/AMLNet.csv`. Check the filename and folder exactly |
| MD5 mismatch at step 00 | Incomplete or corrupted download — download again from Zenodo |
| `MemoryError` during step 01 or 03 | Close other applications. The `metadata` column alone is around 600 MB and is dropped early, so memory pressure is highest in the first stages |
| Step 05 seems to hang | Normal — this is the heaviest stage and it prints only at stage boundaries |
| Step 05 crashes or the machine freezes | Memory exhaustion. Do not raise `N_ITER` or remove the tuning subsample on a 16 GB machine; see the note above |
| Dashboard shows empty tabs | Steps 06–08 have not been run. The dashboard reads their saved outputs and does not compute anything itself |
| `ModuleNotFoundError` | The virtual environment is not active, or `pip install -r requirements.txt` was skipped |

## Repository structure

```
.
├── data/
│   ├── raw/                     AMLNet.csv  (not committed - download from Zenodo)
│   └── processed/               generated by the pipeline (Parquet)
├── dashboard/
│   └── app.py                   Streamlit reviewer dashboard (self-contained)
├── src/                         pipeline source code (see table below)
├── models/                      trained model files + final_model_config.json
├── outputs/
│   ├── figures/                 13 PNG figures
│   └── tables/                  58 CSV result tables
├── README.md
├── requirements.txt
└── .gitignore
```

The whole project is developed and run with the scripts in `src/` (in PyCharm);
there is no separate notebook version of the code.

### `src/` — every file explained

| File | CRISP-DM stage | What it does |
|---|---|---|
| `amlnet_common.py` | shared | Project paths (resolved at runtime, never hard-coded), `RANDOM_STATE = 42`, `set_seeds()`, the leakage/identifier column constants, feature-engineering definitions, metric helpers and `environment_summary()` |
| `amlnet_resampling.py` | shared | The custom SMOTENC pipeline. Orders steps as scale → integer-code categoricals → resample → one-hot, because the standard ordering would hand SMOTENC dummy columns and break its majority-vote step |
| `step00_setup.py` | setup | Verifies the environment, pins and reports package versions, creates the folder structure, and checks `AMLNet.csv` against the published MD5 before anything else runs |
| `step01_data_understanding.py` | Data Understanding | Profiles the dataset — shape, dtypes, missing values, duplicates, class distribution, categorical levels, temporal coverage — and performs the leakage audit |
| `step02_data_preparation.py` | Data Preparation | Drops the seven leakage-prone columns and enforces the exclusion with a runtime assertion; removes the record with a missing target |
| `step03_feature_engineering.py` | Data Preparation | Builds the eleven engineered features and validates that none contain missing or infinite values |
| `step04_split.py` | Data Preparation | Stratified 70/15/15 split at seed 42, with an integrity check on row totals, index disjointness and positive-class presence in every split |
| `step05_model_development.py` | Modelling | Trains Logistic Regression, Random Forest and XGBoost under both imbalance strategies and tunes all six with `RandomizedSearchCV` scored on PR-AUC |
| `step06_threshold_optimization.py` | Evaluation | Sweeps thresholds 0.01–0.99 on validation data, picks the F1-optimal point per candidate, ranks by PR-AUC and writes the winner to `models/final_model_config.json` |
| `step07_final_testing.py` | Evaluation | Loads the frozen model and threshold and evaluates them **once** on the untouched test set |
| `step08_explainability.py` | Evaluation | Computes built-in gain importance, permutation importance and SHAP values, aggregates them to feature groups and generates local explanations |
| `app.py` | Deployment | Reviewer dashboard (there is also a self-contained copy in `dashboard/app.py`) |
| `run_all.py` | orchestration | Runs `step00` to `step08` in order as a single command |

### `outputs/figures/` — 13 figures

`fig01_target_class_distribution`, `fig02_amount_distribution_by_class`,
`fig03_laundering_rate_by_category`, `fig03_model_comparison`,
`fig04_validation_pr_curve`, `fig05_validation_roc_curve`,
`fig06_threshold_optimisation`, `fig07_test_confusion_matrix`,
`fig08_test_pr_curve`, `fig09_shap_global_importance`, `fig10_builtin_importance`,
`fig11_permutation_importance`, `fig12_local_shap_example`

### `outputs/tables/` — 58 CSV tables

| Group | Contents |
|---|---|
| Dataset and profiling | Shape, column profile, duplicates, target distribution, categorical levels, temporal coverage |
| Leakage and cleaning | Leakage audit evidence, cleaning summary, feature decisions |
| Feature engineering | Engineered feature definitions, missing and infinite value checks |
| Split | Train/validation/test statistics and integrity check |
| Modelling | Hyperparameter search results for all six configurations |
| Validation | Model comparison at default and optimised thresholds, threshold sweep |
| Final testing | Test metrics, confusion matrix, reviewer workload, validation-to-test consistency |
| Explainability | Built-in, permutation and SHAP importances, cross-method agreement, local case explanations |

## The reviewer dashboard (`dashboard/app.py`)

```bash
streamlit run dashboard/app.py
```

The dashboard is the deployment prototype for this project. Its key design
decision is that it **reads only the saved outputs of steps 06–08** — the
metrics tables, figures, frozen prediction file and explainability tables —
rather than re-running the model. That means it starts instantly, and the
numbers it displays are the same numbers reported above, because there is no
separate inference path that could show different results. The copy in
`dashboard/app.py` is self-contained (it does not import any pipeline code),
so it can be run on its own.

### Seeing the dashboard without running anything

A **recorded walkthrough of the complete dashboard** is available here:

> **[▶ Dashboard demonstration video](https://drive.google.com/file/d/10jZbtqbLWENUH5QFAOESdcQHTcB09NLs/view)**

The recording shows every tab in use on the full test set, and is the
recommended way to review the dashboard.

**Note on running it locally.** The dashboard reads a prediction file,
`data/processed/test_predictions.parquet`, which is produced by step 07. That
file is not included in this repository because it is too large for GitHub,
in the same way as the raw dataset. Two consequences follow:

- Running the pipeline first (`python src/run_all.py`) generates it, after
  which every tab works
- Opening the dashboard without running the pipeline leaves the **Alert queue**
  and **Case review** tabs empty; the other tabs — Overview, Explainability,
  Risk profile, Performance and Limitations — read the committed CSV tables in
  `outputs/tables/` and work immediately

Every number shown in the dashboard also appears in `outputs/tables/`, which
**is** committed, so all reported results can be verified from this repository
without running or installing anything.

It presents seven tabs:

| Tab | Purpose |
|---|---|
| **Overview** | Headline test metrics, alert count, true and false positives at a glance |
| **Alert queue** | The prioritised list of flagged transactions a reviewer would work through |
| **Case review** | Individual transaction detail with its risk score, band and local SHAP explanation |
| **Explainability** | What drives the model globally — built-in, permutation and SHAP importances side by side |
| **Risk profile** | The characteristics of flagged transactions as a group |
| **Performance** | Confusion matrix, precision–recall curve, threshold optimisation and the automatic model-selection comparison |
| **Limitations** | Synthetic-data caveats and responsible-use guidance, shown in the tool itself rather than buried in a report |

The last tab is deliberate. The dashboard is a technical prototype for
academic demonstration, not a production AML system, and it states that where
a user will actually see it.

## Methodology overview

A CRISP-DM informed experimental lifecycle across six phases.

**Leakage control.** Seven columns are excluded, and the exclusion is enforced
by a runtime assertion so it cannot silently regress:

| Column | Reason |
|---|---|
| `laundering_typology` | Typology label directly reveals the target |
| `metadata` | Generator dictionary with embedded risk scores; also ~600 MB of the 691 MB file |
| `fraud_probability` | Generated risk score (mean 0.025 normal vs 0.405 laundering) |
| `isFraud` | Related generated label, 1:1 with the target |
| `nameOrig`, `nameDest` | High-cardinality account identifiers — memorisation risk |
| `step` | Documented as a sequential transaction step/ID, not a temporal feature; redundant given `hour`, `day_of_week`, `day_of_month`, `month` |

**Feature engineering.** Eleven features derived only from `amount`,
`oldbalanceOrg` and `newbalanceOrig` — balance movement, amount-to-balance
ratios, zero-balance indicators and log transforms — giving 20 model inputs.

**Imbalance handling.** Two paradigms compared under identical tuning: class
weighting (`scale_pos_weight` = 623.996) and SMOTENC oversampling
(`sampling_strategy` = 0.05, `k` = 5). SMOTENC rather than SMOTE because two
features are nominal; the pipeline order is scale → integer-code → resample →
one-hot so the majority-vote step sees real categories rather than dummy
columns.

**Models.** Logistic Regression, Random Forest and XGBoost, each under both
imbalance strategies, tuned with `RandomizedSearchCV` (6 candidates, 3-fold
stratified CV, scored on `average_precision`).

**Selection.** Thresholds swept 0.01–0.99 on validation data with the
F1-maximising point chosen per candidate; models then ranked by PR-AUC with
ties broken by F1 and alert volume. Nothing is chosen by hand — the winner,
its hyperparameters and its threshold are written to
`models/final_model_config.json` and read by the testing, explainability and
dashboard stages, so the deployed artefact is the one that was evaluated.

**Explainability.** Three complementary methods — built-in gain importance,
permutation importance and SHAP TreeExplainer — cross-checked at
feature-group level, plus local explanations for individual flagged
transactions.

## Reproducibility

- Global seed `42` fixed across `random`, `numpy`, `PYTHONHASHSEED`,
  scikit-learn and XGBoost
- All package versions pinned in `requirements.txt`
- Dataset verified by MD5 before use
- Preprocessing fitted on training folds only; validation and test are
  transformed, never fitted
- Test set read exactly once, after model and threshold selection are complete
- Every figure and table regenerated programmatically from saved outputs

## Limitations

AMLNet is synthetic. Laundering-positive cases concentrate heavily in specific
transaction categories by construction of the generator, so the model may
partly be learning generator rules rather than behavioural signal — the
explainability analysis exposes this rather than hiding it. The split is
stratified random rather than chronological and account-disjoint, so it does
not reproduce the deployment case where a model scores future transactions from
unseen accounts. Probability calibration was not assessed, so the dashboard
risk bands are ranking statements rather than verified likelihoods. The
dashboard has not been evaluated with practising AML analysts.

**This is academic research, not a production AML system.** It does not replace
human judgement or compliance sign-off.

## Citation

Dataset:

> Huda, S. (2025). *AMLNet: Synthetic anti-money laundering transaction
> dataset.* Zenodo. https://doi.org/10.5281/zenodo.16736515

## Licence

Code in this repository is provided for academic use. The AMLNet dataset is
licensed CC BY-NC 4.0 and is used here strictly non-commercially with
attribution to its author.