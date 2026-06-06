
# Data Familiarization

## Exploration Context

**Dataset:** Heart Disease Prediction Kaggle competition data
**Source:** `/Users/temicide/Documents/5_domain_final/Heart-Disease/data/super-ai-engineer-ss-6-individual-heart-disease-prediction`
**Exploration Goal:** discover patterns and quirks that can help score the Kaggle submission.

## Tables Overview

| table | rows | columns | apparent purpose | grain |
|---|---:|---:|---|---|
| `train_raw` | 223,084 | 20 | training survey records with target | one respondent |
| `train_labeled` | 221,390 | 20 | supervised subset after removing missing/blank target | one labeled respondent |
| `test_raw` | 74,361 | 19 | competition test records without target | one respondent |
| `sample_submission_raw` | 74,361 | 2 | required submission shape | one test ID |

## Schema Details

```sql
PRAGMA table_info(train_raw);
PRAGMA table_info(test_raw);
PRAGMA table_info(sample_submission_raw);
```

| table | column | dtype | non_null | missing | unique_values |
| --- | --- | --- | --- | --- | --- |
| train_raw | ID | object | 223084 | 0 | 223084 |
| train_raw | History of HeartDisease or Attack | object | 221390 | 1694 | 2 |
| train_raw | High Blood Pressure | object | 223084 | 0 | 2 |
| train_raw | Told High Cholesterol | object | 190898 | 32186 | 2 |
| train_raw | Cholesterol Checked | object | 223084 | 0 | 2 |
| train_raw | Body Mass Index | float64 | 211302 | 11782 | 4907 |
| train_raw | Smoked 100+ Cigarettes | object | 223083 | 1 | 2 |
| train_raw | Diagnosed Stroke | object | 223084 | 0 | 2 |
| train_raw | Diagnosed Diabetes | object | 223081 | 3 | 2 |
| train_raw | Leisure Physical Activity | object | 223084 | 0 | 2 |
| train_raw | Heavy Alcohol Consumption | object | 223084 | 0 | 2 |
| train_raw | Health Care Coverage | object | 223084 | 0 | 2 |
| train_raw | Doctor Visit Cost Barrier | object | 223083 | 1 | 2 |
| train_raw | General Health | object | 223083 | 1 | 5 |
| train_raw | Difficulty Walking | object | 223081 | 3 | 2 |
| train_raw | Sex | object | 223084 | 0 | 2 |
| train_raw | Education Level | object | 223084 | 0 | 6 |
| train_raw | Income Level | object | 223084 | 0 | 8 |
| train_raw | Age | int64 | 223084 | 0 | 83 |
| train_raw | Vegetable or Fruit Intake (1+ per Day) | object | 223084 | 0 | 2 |
| test_raw | ID | object | 74361 | 0 | 74361 |
| test_raw | High Blood Pressure | object | 74361 | 0 | 2 |
| test_raw | Told High Cholesterol | object | 74361 | 0 | 2 |
| test_raw | Cholesterol Checked | object | 74361 | 0 | 2 |
| test_raw | Body Mass Index | float64 | 74361 | 0 | 3921 |
| test_raw | Smoked 100+ Cigarettes | object | 74361 | 0 | 2 |
| test_raw | Diagnosed Stroke | object | 74361 | 0 | 2 |
| test_raw | Diagnosed Diabetes | object | 74361 | 0 | 2 |
| test_raw | Leisure Physical Activity | object | 74361 | 0 | 2 |
| test_raw | Heavy Alcohol Consumption | object | 74361 | 0 | 2 |
| test_raw | Health Care Coverage | object | 74361 | 0 | 2 |
| test_raw | Doctor Visit Cost Barrier | object | 74361 | 0 | 2 |
| test_raw | General Health | object | 74361 | 0 | 5 |
| test_raw | Difficulty Walking | object | 74361 | 0 | 2 |
| test_raw | Sex | object | 74361 | 0 | 2 |
| test_raw | Education Level | object | 74361 | 0 | 6 |
| test_raw | Income Level | object | 74361 | 0 | 8 |
| test_raw | Age | int64 | 74361 | 0 | 83 |
| test_raw | Vegetable or Fruit Intake (1+ per Day) | object | 74361 | 0 | 2 |
| sample_submission_raw | ID | object | 74361 | 0 | 74361 |
| sample_submission_raw | History of HeartDisease or Attack | object | 3 | 74358 | 1 |

**Potential relationships:** `sample_submission_raw.ID` should align one-to-one and in order with `test_raw.ID`; no other join tables are present.

**Temporal coverage:** no date/time column exists. `Age` is an ordered respondent attribute, not a collection timestamp.

## Data Quality Assessment

### Target Balance

```sql
SELECT
  "History of HeartDisease or Attack" AS target_value,
  COUNT(*) AS rows
FROM train_raw
GROUP BY target_value;
```

| target_value | rows | pct_train | pct_labeled |
| --- | --- | --- | --- |
| No | 203322 | 91.14% | 91.84% |
| Yes | 18068 | 8.10% | 8.16% |
| Missing | 1694 | 0.76% |  |

### Completeness

```sql
SELECT COUNT(*) AS rows FROM train_raw;
-- Per-column missingness computed across train_raw and test_raw.
```

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

### Numeric Ranges

```sql
SELECT
  MIN(Age) AS min_age,
  MAX(Age) AS max_age,
  MIN("Body Mass Index") AS min_bmi,
  MAX("Body Mass Index") AS max_bmi
FROM train_raw;
```

| column | count | missing | min | p01 | p25 | median | p75 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Age | 223084 | 0 | 18.00 | 19.00 | 42.00 | 56.00 | 67.00 | 97.00 | 100.00 | 54.74 |
| Body Mass Index | 211302 | 11782 | 11.21 | 17.94 | 23.88 | 27.07 | 31.10 | 49.56 | 98.63 | 28.19 |

### ID and Submission Integrity

| check | result |
|---|---:|
| duplicate train IDs | 0 |
| duplicate test IDs | 0 |
| train/test ID overlap | 0 |
| sample rows matching test rows | True |
| blank sample target values | 74,358 |

## Initial Impressions

- Data quality is good enough for modeling after dropping only missing target rows for supervised training.
- Predictor missingness is asymmetric: train has missing features, while test has no missing features.
- The base positive class is rare (8.16%), so accuracy alone is misleading.
- Many features are categorical strings; treating ordinal categorical values as unordered may leave signal on the table.

## Exploration Strategy

1. **Time-based patterns:** no true temporal features; inspect age-ordered risk and ID-order stability as the closest available ordered dimensions.
2. **Segmentation patterns:** compare target rates across clinical, demographic, socioeconomic, access, and lifestyle segments.
3. **Relationship patterns:** inspect correlations, risk-count features, interactions, and train/test distribution drift.
