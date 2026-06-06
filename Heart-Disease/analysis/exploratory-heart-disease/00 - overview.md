
# Heart Disease Kaggle EDA Overview

## Context

Dataset: `super-ai-engineer-ss-6-individual-heart-disease-prediction`

Source: local files in `/Users/temicide/Documents/5_domain_final/Heart-Disease/data/super-ai-engineer-ss-6-individual-heart-disease-prediction`.

Goal: find target patterns and data quirks that can improve a Kaggle submission for `History of HeartDisease or Attack`.

SQLite workspace: `/Users/temicide/Documents/5_domain_final/Heart-Disease/analysis/exploratory-heart-disease/heart_disease.sqlite` with `train_raw`, `test_raw`, `sample_submission_raw`, and `train_labeled`.

## Exploration Summary

- Train has 223,084 rows, test has 74,361 rows, and sample submission has 74,361 rows.
- Usable supervised train rows: 221,390. Missing/blank target rows: 1,694.
- Labeled positive rate: 8.16% (18,068 `Yes`, 203,322 `No`).
- No true date/time column exists. The temporal phase therefore checks the absence of temporal coverage, age-ordered risk, and ID-order stability.
- Strongest univariate signals are stroke history, poor general health, difficulty walking, diabetes, blood pressure, cholesterol, age, sex, income, and education.
- Most important data quality issue for scoring: train-only missingness in `Told High Cholesterol`, `Body Mass Index`, and target; test has no missing feature values.

## Highest-Value Submission Ideas

1. Use GBDT models with native/robust categorical handling; the data is mixed tabular, imbalanced, and nonlinear.
2. Add interaction/risk-count features: clinical risk count, cardiometabolic cluster, general-health x walking difficulty, age x high blood pressure, BMI class, ordered income/education.
3. Preserve missingness in training but avoid over-trusting train-only missing flags because test has no missing feature values.
4. Tune the final label threshold from out-of-fold predictions. The base positive rate is only 8.16%, so default `0.5` probability threshold is unlikely to be optimal for F1-like metrics.
5. Validate improvements with stratified OOF metrics before using public leaderboard feedback.
