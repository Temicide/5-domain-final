# Heart Disease Prediction Competition Spec

Competition: https://www.kaggle.com/competitions/super-ai-engineer-ss-6-individual-heart-disease-prediction

## Goal

Build the strongest possible Google Colab-ready solution for predicting `History of HeartDisease or Attack` from tabular health survey features. The target labels are string classes: `Yes` and `No`.

The local data strongly resembles BRFSS-style population survey data: binary health history fields, self-reported general health, demographics, income, education, BMI, and age. This should be treated as an imbalanced binary tabular classification problem.

## Colab Notebook Requirement

The deliverable must be a Google Colab-ready notebook solution. The notebook must:

- Run end-to-end in Google Colab from a fresh runtime.
- Install or import required packages in notebook cells.
- Authenticate to Kaggle inside Colab using either an uploaded `kaggle.json` file or Colab secrets/environment variables.
- Never print Kaggle credentials, file contents from `kaggle.json`, or secret values.
- Use the Kaggle CLI or Kaggle API inside Colab to download the competition data before reading any competition CSVs.
- Extract the downloaded competition archive into a Colab input directory such as `/content/input/super-ai-engineer-ss-6-individual-heart-disease-prediction`.
- Read competition files from `/content/input/...` after extraction when running in Colab.
- Use `/content/working` for runtime artifacts and generate `/content/submission.csv`.
- Validate the generated CSV before declaring the run complete.
- Keep local development fallbacks under `/Users/temicide/Documents/5_domain_final/Heart-Disease`.
- Not submit through the Kaggle API, Kaggle CLI, or any other automated submission mechanism.

Manual upload of `/content/submission.csv` to Kaggle is allowed, but implementation must stop after creating and validating the CSV.

## Kaggle Data Access In Colab

The notebook must support both credential flows below without exposing secrets:

1. Uploaded `kaggle.json`:
   - Prompt the user to upload `kaggle.json` with `google.colab.files.upload()`.
   - Save it to `/root/.kaggle/kaggle.json`.
   - Set file permissions to `600`.
   - Do not display the uploaded JSON content.

2. Colab secrets or environment variables:
   - Read `KAGGLE_USERNAME` and `KAGGLE_KEY` from Colab secrets or environment variables.
   - Export them only into the current process environment.
   - Do not print either value.

After credentials are available, data download must run before CSV loading, for example:

```bash
kaggle competitions download -c super-ai-engineer-ss-6-individual-heart-disease-prediction -p /content/input
```

Then extract the archive to:

```text
/content/input/super-ai-engineer-ss-6-individual-heart-disease-prediction
```

Accept either Kaggle CLI or `kaggle.api.kaggle_api_extended.KaggleApi` as long as the notebook performs the same download-and-extract workflow in Colab.

## Kaggle Context

The competition page is a JavaScript app and unauthenticated page access may not expose the evaluation metric. Before final manual upload, verify the official metric in the Kaggle UI. Until then, optimize models in a metric-safe way:

- Primary ranking metrics: ROC-AUC and PR-AUC.
- Threshold metrics: F1, balanced accuracy, and accuracy.
- Submission labels: convert probabilities to `Yes`/`No` using a validation-tuned threshold if the official metric is F1/accuracy-like; use calibrated ranking probabilities only internally.

The notebook must use Kaggle only as the competition data source and optional manual upload destination. It must not automate competition submission.

## Data Locations

Colab paths:

- Input root: `/content/input`
- Competition data directory: `/content/input/super-ai-engineer-ss-6-individual-heart-disease-prediction`
- Working directory: `/content/working`
- Required submission path: `/content/submission.csv`
- Optional copied artifact path: `/content/working/submission.csv`

Local development fallbacks:

- Project root: `/Users/temicide/Documents/5_domain_final/Heart-Disease`
- Local data directory: `/Users/temicide/Documents/5_domain_final/Heart-Disease/data/super-ai-engineer-ss-6-individual-heart-disease-prediction`
- Local output directory: `/Users/temicide/Documents/5_domain_final/Heart-Disease/outputs`
- Local submission fallback: `/Users/temicide/Documents/5_domain_final/Heart-Disease/outputs/submissions/submission.csv`

Path resolution should prefer Colab data only after the Kaggle download and extraction have produced the expected CSV files. If Colab paths are unavailable, use the local development fallback.

## Local Data Audit

Files:

- `data/super-ai-engineer-ss-6-individual-heart-disease-prediction/train.csv`
- `data/super-ai-engineer-ss-6-individual-heart-disease-prediction/test.csv`
- `data/super-ai-engineer-ss-6-individual-heart-disease-prediction/sample_submission.csv`

Shapes:

- Train: 223,084 rows, 20 columns.
- Test: 74,361 rows, 19 columns.
- Sample submission: 74,361 rows, 2 columns.

Target:

- Target column: `History of HeartDisease or Attack`.
- Usable labeled train rows: 221,390.
- Missing target rows: 1,694. Do not train supervised models on these rows unless using semi-supervised experiments.
- Positive class rate among labeled rows: 8.16% `Yes`.
- This imbalance makes naive accuracy misleading.

Important file quirks:

- CSVs contain a UTF-8 BOM on the `ID` column; read with `encoding="utf-8-sig"`.
- `sample_submission.csv` has many blank target values locally. Generate a complete submission with one row per test ID and no blanks.

Missing values:

- Train has missing feature values; test has no missing values.
- `Told High Cholesterol`: 32,186 missing in train, 0 missing in test.
- `Body Mass Index`: 11,782 missing in train, 0 missing in test.
- Target missing: 1,694 rows.
- Minor missing values also appear in diabetes, walking difficulty, general health, smoker, and cost barrier.

Do not silently drop rows with missing predictors. Keep missingness as signal for categorical variables and use robust numeric imputation for BMI.

## Strongest Signals From EDA

Univariate target rates among labeled rows:

- `Diagnosed Stroke=Yes`: 37.7% positive.
- `General Health=Very Poor`: 32.0% positive.
- `Difficulty Walking=Yes`: 21.8% positive.
- `Diagnosed Diabetes=Yes`: 20.5% positive.
- `General Health=Poor`: 19.0% positive.
- `High Blood Pressure=Yes`: 15.7% positive.
- `Told High Cholesterol=Yes`: 15.4% positive.
- Male: 10.5% positive vs Female: 6.3%.
- Lower education and lower income have higher positive rates.
- Age and BMI are continuous predictors and should be modeled nonlinearly.

Train/test drift:

- Most features have modest train/test distribution shift.
- Largest shift is from train-only missingness in `Told High Cholesterol` and BMI. Treat missingness carefully; do not overfit to missingness if the final test has no missing values.

## Evidence From Literature And Benchmarks

Tabular ML benchmarks and recent surveys continue to show gradient-boosted decision trees as the default strongest approach for medium-sized heterogeneous tabular data, especially against generic neural networks. Recent tabular papers also note that strong pre-tuned MLPs can compete, but GBDTs remain the pragmatic first choice for Kaggle-style structured data.

Heart disease and BRFSS-related studies specifically report strong performance from ensemble/tree methods such as Gradient Boosting, XGBoost, Random Forest, and LightGBM. For this dataset size and feature mix, prioritize tuned GBDT ensembles over deep learning unless the leaderboard stalls.

Useful references:

- Deep Neural Networks and Tabular Data: A Survey, IEEE TNNLS 2022/2024 indexing.
- Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data, NeurIPS 2024.
- Predicting cardiovascular diseases using imbalanced data: An XGBoost-based analysis of the 2022 BRFSS dataset, 2026.
- Interpretable LightGBM model for predicting coronary heart disease, 2025.

## Local Baseline Results

Three-fold stratified CV on the 221,390 labeled rows:

| Model | ROC-AUC | PR-AUC | Best F1 | Best F1 Threshold | Best Accuracy |
|---|---:|---:|---:|---:|---:|
| Logistic regression, one-hot, class-balanced | 0.8599 | 0.3596 | 0.3594 | 0.500 | 0.7665 |
| LightGBM native categorical | 0.8623 | 0.3650 | 0.4163 | 0.195 | 0.9201 |
| LightGBM native categorical, weighted | 0.8611 | 0.3646 | 0.3569 | 0.500 | 0.7607 |

Interpretation:

- Native categorical LightGBM is already stronger than logistic regression.
- Weighting positives is not automatically helpful; it hurts F1 in this quick benchmark.
- If the official metric is F1, threshold tuning matters more than class weighting.
- If the official metric is accuracy, the threshold will likely be much higher and may predict relatively few positives.
- If the official metric is ROC-AUC/log loss, generate probability-derived labels only if the competition requires labels; otherwise use probabilities if allowed by the rules.

## Validation Protocol

Use this exact protocol before trusting leaderboard improvements:

1. Download and extract competition data in Colab before loading CSVs.
2. Drop only rows with missing target.
3. Use `StratifiedKFold`, 5 folds, fixed seeds.
4. Track ROC-AUC, PR-AUC, F1, accuracy, balanced accuracy, precision, recall, and confusion matrix.
5. Tune thresholds out-of-fold only. Never tune threshold on public leaderboard feedback.
6. Repeat final CV with at least 3 random seeds for top candidates.
7. Compare every experiment against the current OOF baseline, not just public LB.
8. Save OOF predictions for every serious model for later ensembling.

Suggested acceptance gates:

- Any new feature/model must improve 5-fold OOF ROC-AUC or PR-AUC, or improve official-metric proxy after threshold tuning.
- A leaderboard-only gain without OOF support is suspect.
- A valid Colab run must create `/content/submission.csv` and pass submission validation.

## Modeling Ladder

### Phase 1: Reproducible Colab Baseline

Build one clean pipeline:

- Install/import Kaggle access dependencies in Colab.
- Resolve Kaggle credentials from uploaded `kaggle.json` or secrets/environment variables without printing them.
- Download the competition archive via Kaggle CLI or API.
- Extract data under `/content/input/super-ai-engineer-ss-6-individual-heart-disease-prediction`.
- Read CSVs with `utf-8-sig`.
- Rename/normalize `ID` safely.
- Drop missing-target rows for supervised training.
- Preserve categorical strings.
- Impute BMI with median or leave numeric missing for tree models that support missing values.
- Train LightGBM with categorical handling or a robust encoded fallback.
- Tune threshold on OOF predictions.
- Write and validate `/content/submission.csv`.

### Phase 2: Core GBDT Models

Train and save OOF/test predictions from:

- LightGBM.
- CatBoost with categorical features.
- XGBoost using ordinal/category encoding.
- ExtraTrees/RandomForest as a diversity model.
- Logistic regression one-hot as a calibration/diversity model.

Tune each with Optuna or a small hand search. Prioritize:

- `num_leaves`, `max_depth`, `min_child_samples`.
- `learning_rate`, `n_estimators`.
- `subsample`, `colsample_bytree`.
- `reg_alpha`, `reg_lambda`.
- Class imbalance knobs only as experiments, not defaults.

### Phase 3: Feature Engineering

High-value features to test:

- Age bins: decade bins, elderly flags, spline-like bins.
- BMI bins: underweight, normal, overweight, obese class I/II/III.
- Risk count: sum of high blood pressure, high cholesterol, smoker, stroke, diabetes, difficulty walking, poor health.
- Cardiometabolic cluster: blood pressure + cholesterol + diabetes + BMI.
- Lifestyle protective count: physical activity, fruit/vegetable intake, no heavy alcohol.
- Socioeconomic rank: ordered education and income encodings.
- Health access friction: no coverage or doctor visit cost barrier.
- Interactions:
  - age x sex
  - age x blood pressure
  - age x cholesterol
  - age x diabetes
  - stroke x walking difficulty
  - general health x walking difficulty
  - income x health care coverage

Keep engineered features simple and auditable. GBDTs will discover many interactions, but explicit risk counts often help with low-cardinality health survey data.

### Phase 4: Ensembling

Create an ensemble only after individual OOF files exist:

- Average rank-normalized probabilities from top diverse models.
- Optimize blend weights on OOF predictions, constrained to avoid overfitting.
- Try simple mean, weighted mean, and logistic stacker.
- Calibrate final probabilities with isotonic or Platt scaling only if the metric or thresholding benefits.

For F1-like metrics:

- Tune one global threshold on blended OOF predictions.
- Also test fold-specific thresholds for diagnostics, but generate the final `submission.csv` with one global threshold unless evidence is strong.

For accuracy-like metrics:

- Tune threshold for max OOF accuracy.
- Because the positive class is only 8.16%, expect the best accuracy threshold to be conservative.

## Experiment Backlog

Run experiments in this order:

1. Confirm official Kaggle metric from UI.
2. Implement Colab credential handling and Kaggle download/extract workflow.
3. Implement baseline LightGBM 5-fold CV and `/content/submission.csv` generation.
4. Add threshold tuning and report all threshold metrics.
5. Add age/BMI bins and risk-count features.
6. Train CatBoost and compare OOF.
7. Train XGBoost and compare OOF.
8. Build simple average ensemble.
9. Build constrained weighted ensemble.
10. Test calibration and threshold stability.
11. Generate candidate CSVs only for models that improve OOF or have a clear diversity reason.

## Submission Requirements

Submission file:

- Columns: `ID`, `History of HeartDisease or Attack`.
- Row count: 74,361.
- IDs must match test IDs exactly.
- Target values must be only `Yes` or `No`.
- No blank values.
- Required output path in Colab: `/content/submission.csv`.
- Optional mirrored output path in Colab: `/content/working/submission.csv`.
- Local fallback output path: `/Users/temicide/Documents/5_domain_final/Heart-Disease/outputs/submissions/submission.csv`.
- Do not call `kaggle competitions submit`, `KaggleApi.competition_submit`, or any automated submission equivalent.
