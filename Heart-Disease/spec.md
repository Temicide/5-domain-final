# Heart Disease Prediction Competition Spec

Competition: https://www.kaggle.com/competitions/super-ai-engineer-ss-6-individual-heart-disease-prediction

## Goal

Build the strongest possible Kaggle submission for predicting `History of HeartDisease or Attack` from tabular health survey features. The target labels are string classes: `Yes` and `No`.

The local data strongly resembles BRFSS-style population survey data: binary health history fields, self-reported general health, demographics, income, education, BMI, and age. This should be treated as an imbalanced binary tabular classification problem.

## Kaggle Notebook Requirement

The deliverable must be a Kaggle Notebook ready solution. The notebook must:

- Run end-to-end in the Kaggle notebook environment.
- Read competition files from `/kaggle/input/...` when running on Kaggle, with local path fallbacks only for development.
- Generate `/kaggle/working/submission.csv`.
- Validate the generated CSV before saving or before declaring the run complete.
- Not submit through the Kaggle API, Kaggle CLI, or any other automated submission mechanism.

Manual upload or normal Kaggle notebook output use is allowed, but implementation must stop after generating `submission.csv`.

## Kaggle Context

The local Kaggle skill credential check found legacy Kaggle credentials:

- `KAGGLE_USERNAME` and `KAGGLE_KEY` are available.
- `KAGGLE_API_TOKEN` is missing.
- The `kaggle` CLI binary is not installed in this environment.

The Kaggle competition page is a JavaScript app and the unauthenticated page shell did not expose the evaluation metric through `curl` or web search. Before final submission, verify the official metric in the Kaggle UI. Until then, optimize models in a metric-safe way:

- Primary ranking metrics: ROC-AUC and PR-AUC.
- Threshold metrics: F1, balanced accuracy, and accuracy.
- Submission labels: convert probabilities to `Yes`/`No` using a validation-tuned threshold if the official metric is F1/accuracy-like; use calibrated ranking probabilities only internally.

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

1. Drop only rows with missing target.
2. Use `StratifiedKFold`, 5 folds, fixed seeds.
3. Track ROC-AUC, PR-AUC, F1, accuracy, balanced accuracy, precision, recall, and confusion matrix.
4. Tune thresholds out-of-fold only. Never tune threshold on public leaderboard feedback.
5. Repeat final CV with at least 3 random seeds for top candidates.
6. Compare every experiment against the current OOF baseline, not just public LB.
7. Save OOF predictions for every serious model for later ensembling.

Suggested acceptance gates:

- Any new feature/model must improve 5-fold OOF ROC-AUC or PR-AUC, or improve official-metric proxy after threshold tuning.
- A leaderboard-only gain without OOF support is suspect.

## Modeling Ladder

### Phase 1: Reproducible Baseline

Build one clean pipeline:

- Read CSVs with `utf-8-sig`.
- Rename/normalize `ID` safely.
- Drop missing-target rows for supervised training.
- Preserve categorical strings.
- Impute BMI with median or leave numeric missing for tree models that support missing values.
- Train LightGBM with native categorical features.
- Tune threshold on OOF predictions.
- Write a complete submission.

### Phase 2: Core GBDT Models

Train and save OOF/test predictions from:

- LightGBM native categorical.
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
2. Implement baseline LightGBM native categorical 5-fold CV and submission.
3. Add threshold tuning and report all threshold metrics.
4. Add age/BMI bins and risk-count features.
5. Train CatBoost and compare OOF.
6. Train XGBoost and compare OOF.
7. Build simple average ensemble.
8. Build constrained weighted ensemble.
9. Test calibration and threshold stability.
10. Generate leaderboard-candidate CSVs only for models that improve OOF or have a clear diversity reason.

## Submission Requirements

Submission file:

- Columns: `ID`, `History of HeartDisease or Attack`.
- Row count: 74,361.
- IDs must match test IDs exactly.
- Target values must be only `Yes` or `No`.
- No blank values.
- Required output path on Kaggle: `/kaggle/working/submission.csv`.
- Do not call Kaggle API/CLI submission commands from the notebook.

Use this final conversion pattern:

```python
submission = pd.DataFrame({
    "ID": test["ID"],
    "History of HeartDisease or Attack": np.where(test_pred >= threshold, "Yes", "No"),
})
submission.to_csv("submission.csv", index=False)
```

## Current Best Plan

The most likely path to a strong score is:

1. Verify official metric.
2. Build 5-fold LightGBM native categorical baseline.
3. Tune threshold if the metric uses hard labels.
4. Add compact domain/risk-count features.
5. Add CatBoost and XGBoost for diversity.
6. Blend OOF-validated predictions.
7. Avoid public leaderboard overfitting; use leaderboard only as final confirmation.

Based on local evidence, plain native-categorical LightGBM is the immediate baseline to beat: ROC-AUC `0.8623`, PR-AUC `0.3650`, best local F1 `0.4163` at threshold `0.195`.
