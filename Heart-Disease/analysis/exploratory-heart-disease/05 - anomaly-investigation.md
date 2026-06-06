
# Anomaly Investigation

## Anomalies Identified

1. **Train-only feature missingness** - found in data familiarization and train/test drift.
2. **Missing/blank target rows** - found in target balance profiling.
3. **Blank local sample submission targets** - found in submission integrity checks.
4. **No temporal fields** - found in temporal exploration.
5. **Very high-risk but small clinical intersections** - found in segmentation cross-segments.

## Anomaly 1: Train-Only Feature Missingness

### Where Found

`01 - data-familiarization.md` and `04-relationship-patterns.md`.

### Why It's Anomalous

Test has no missing feature values, while train has missing values in important predictors.

### Investigation Query

```sql
SELECT
  COUNT(*) AS train_rows,
  SUM("Told High Cholesterol" IS NULL) AS missing_cholesterol,
  SUM("Body Mass Index" IS NULL) AS missing_bmi
FROM train_raw;
```

### Results

| column | train_missing | train_missing_pct | test_missing | test_missing_pct |
| --- | --- | --- | --- | --- |
| Told High Cholesterol | 32186 | 14.43% | 0.000 | 0.00% |
| Body Mass Index | 11782 | 5.28% | 0.000 | 0.00% |
| History of HeartDisease or Attack | 1694 | 0.76% |  |  |
| Diagnosed Diabetes | 3 | 0.00% | 0.000 | 0.00% |
| Difficulty Walking | 3 | 0.00% | 0.000 | 0.00% |
| General Health | 1 | 0.00% | 0.000 | 0.00% |
| Smoked 100+ Cigarettes | 1 | 0.00% | 0.000 | 0.00% |
| Doctor Visit Cost Barrier | 1 | 0.00% | 0.000 | 0.00% |

### Explanation

**Determination:** data quality/split artifact.

**Reasoning:** The missingness is concentrated in train and absent from test, so a model that leans heavily on missingness flags may overfit train-only data collection artifacts.

**Action:** Preserve missingness during CV, but test model variants with and without missingness flags. For BMI use numeric imputation or tree-native missing support; for categorical missingness use a `Missing` level but monitor feature importance.

## Anomaly 2: Missing/Blank Target Rows

### Where Found

Target balance profiling.

### Why It's Anomalous

1,694 training rows lack supervised labels.

### Investigation Query

```sql
SELECT
  COUNT(*) AS rows,
  SUM("History of HeartDisease or Attack" IS NULL OR trim("History of HeartDisease or Attack") = '') AS missing_target
FROM train_raw;
```

### Results

| metric | value |
|---|---:|
| train rows | 223,084 |
| missing/blank target rows | 1,694 |
| labeled rows | 221,390 |

### Explanation

**Determination:** data quality issue for supervised training.

**Action:** Exclude these rows from supervised model fitting and OOF metrics. They could be used only in a deliberate semi-supervised experiment.

## Anomaly 3: Blank Sample Submission Targets

### Where Found

Submission integrity checks.

### Why It's Anomalous

The local `sample_submission.csv` has the required shape but 74,358 blank target values.

### Investigation Query

```sql
SELECT COUNT(*) AS rows
FROM sample_submission_raw
WHERE "History of HeartDisease or Attack" IS NULL OR trim("History of HeartDisease or Attack") = '';
```

### Results

| metric | value |
|---|---:|
| sample rows | 74,361 |
| blank target values | 74,358 |

### Explanation

**Determination:** expected sample-submission placeholder.

**Action:** A valid generated submission must fill every row with `Yes` or `No` and keep test ID order unchanged.

## Anomaly 4: No Temporal Fields

### Where Found

Temporal exploration.

### Why It's Anomalous

The exploratory-analysis process expects temporal exploration, but the dataset has no time axis.

### Explanation

**Determination:** real limitation of competition files, not an error.

**Action:** Do not invent seasonality. Use age as an ordered health feature, and keep validation stratified rather than time-based.

## Anomaly 5: High-Risk Small Clinical Intersections

### Where Found

Cross-segmentation analysis.

### Why It's Anomalous

Some intersections have very high positive rates but limited sample sizes, increasing overfit risk for target encoding or hand-built rules.

### Results

| interaction | n | positive_rate | lift_vs_base |
| --- | --- | --- | --- |
| Diagnosed Stroke=Yes | General Health=Very Poor | 1611 | 56.18% | 6.88x |
| Diagnosed Stroke=Yes | Diagnosed Diabetes=Yes | 2544 | 48.82% | 5.98x |
| Diagnosed Stroke=Yes | Difficulty Walking=Yes | 3914 | 46.96% | 5.75x |
| Diagnosed Stroke=Yes | General Health=Poor | 2278 | 42.45% | 5.20x |
| Difficulty Walking=Yes | General Health=Very Poor | 7343 | 34.80% | 4.26x |
| Income Level=($10,000 to less than $15,000 | General Health=Very Poor | 1601 | 34.73% | 4.26x |
| Income Level=$20,000 to less than $25,000 | General Health=Very Poor | 1365 | 34.21% | 4.19x |
| Income Level=$25,000 to less than $35,000 | General Health=Very Poor | 1146 | 33.68% | 4.13x |
| Diagnosed Stroke=Yes | Diagnosed Diabetes=No | 5405 | 32.45% | 3.98x |
| Income Level=$15,000 to less than $20,000 | General Health=Very Poor | 1529 | 32.37% | 3.97x |

### Explanation

**Determination:** real phenomenon with modeling caveat.

**Action:** Use interactions as model features, not hard rules. Validate with out-of-fold metrics and avoid target encoding without leakage-safe folds.

## Anomalies Summary

### Real Phenomena

1. No temporal metadata in the competition files.
2. High-risk clinical intersections among stroke, general health, walking difficulty, diabetes, blood pressure, and cholesterol.

### Data Quality Issues

1. Train-only predictor missingness.
2. Missing target rows in train.
3. Blank target values in sample submission.

### Implications for Pattern Discovery

- Missing target rows must be excluded from supervised EDA and modeling.
- Missingness itself may be signal in train, but it is also a drift source because test has no missing predictors.
- Strong clinical interactions are useful candidates for feature engineering but need OOF validation.
