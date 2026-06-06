
# Relationship Pattern Exploration

## Objective

Discover correlations, associations, dependencies, and drift patterns that affect feature engineering and validation.

## Exploration Approach

1. Encode simple clinical/demographic indicators and compute correlation with the target.
2. Derive risk-count features and inspect target rates.
3. Compare train/test distributions for drift.

## Analysis 1: Correlation With Target

### Rationale

Simple correlations are not a final model, but they reveal signal direction and sanity-check feature importance.

### Query

```sql
-- Binary/ordinal encodings were derived from train_labeled, then correlated with target_yes.
SELECT feature, corr_with_target
FROM derived_correlations
ORDER BY ABS(corr_with_target) DESC;
```

### Results

| feature | corr_with_target |
| --- | --- |
| age | 0.231 |
| general_health_rank | 0.225 |
| High Blood Pressure=Yes | 0.219 |
| Difficulty Walking=Yes | 0.215 |
| Diagnosed Stroke=Yes | 0.208 |
| Told High Cholesterol=Yes | 0.197 |
| Diagnosed Diabetes=Yes | 0.176 |
| income_rank | -0.118 |
| Smoked 100+ Cigarettes=Yes | 0.111 |
| Cholesterol Checked=Yes | 0.102 |
| education_rank | -0.077 |
| Leisure Physical Activity=Yes | -0.077 |
| male | 0.076 |
| bmi | 0.057 |
| Health Care Coverage=Yes | 0.034 |
| Heavy Alcohol Consumption=Yes | -0.028 |
| Doctor Visit Cost Barrier=Yes | 0.020 |
| Vegetable or Fruit Intake (1+ per Day)=Yes | -0.019 |

### Observations

- General health rank, difficulty walking, age, stroke, diabetes, high blood pressure, and cholesterol are directionally aligned with higher risk.
- Income and education have negative correlations with target after ordinal encoding: higher socioeconomic rank aligns with lower observed positive rate.

## Analysis 2: Risk Counts

### Rationale

Risk counts compress several sparse categorical predictors into dense features that can help both linear and tree models.

### Query

```sql
SELECT clinical_risk_count, COUNT(*) AS n, AVG(target_yes) AS positive_rate
FROM derived_clinical_risk_counts
GROUP BY clinical_risk_count;
```

### Results: Clinical Risk Count

| clinical_risk_count | n | positive_rate | lift_vs_base | bar |
| --- | --- | --- | --- | --- |
| 0 | 91597 | 1.38% | 0.17x | # |
| 1 | 60285 | 5.29% | 0.65x | #### |
| 2 | 41244 | 13.88% | 1.70x | ########### |
| 3 | 19966 | 23.29% | 2.85x | ################### |
| 4 | 7202 | 36.10% | 4.42x | ############################# |
| 5 | 1096 | 58.49% | 7.17x | ############################################### |

### Results: Lifestyle Protective Count

| lifestyle_protective_count | n | positive_rate | lift_vs_base | bar |
| --- | --- | --- | --- | --- |
| 0 | 654 | 6.57% | 0.81x | ############# |
| 1 | 14183 | 10.89% | 1.33x | ###################### |
| 2 | 66057 | 10.09% | 1.24x | #################### |
| 3 | 140496 | 6.99% | 0.86x | ############## |

### Observations

- Clinical risk count is strongly monotonic and should be a high-priority engineered feature.
- Lifestyle protective count has a weaker but useful gradient.

## Analysis 3: Train/Test Distribution Drift

### Rationale

Submission performance depends on generalizing from train to test. Distribution shifts flag features that may be risky or need careful imputation.

### Query

```sql
-- For categorical features, compare the largest train/test level-frequency difference.
-- For numeric features, compare missingness and mean shift.
SELECT feature, largest_level_or_metric, train_pct, test_pct, abs_diff_pct
FROM derived_train_test_drift
ORDER BY abs_diff_pct DESC;
```

### Results

| feature | largest_level_or_metric | train_pct | test_pct | abs_diff_pct |
| --- | --- | --- | --- | --- |
| Told High Cholesterol | Missing | 14.43% | 0.00% | 14.43% |
| Cholesterol Checked | Yes | 82.38% | 96.29% | 13.91% |
| Body Mass Index | missing_rate | 5.28% | 0.00% | 5.28% |
| High Blood Pressure | Yes | 38.88% | 43.04% | 4.16% |
| Education Level | College graduate | 39.59% | 42.36% | 2.77% |
| Income Level | $75,000 or more | 33.04% | 35.79% | 2.75% |
| Health Care Coverage | No | 7.33% | 4.83% | 2.50% |
| Doctor Visit Cost Barrier | No | 89.90% | 91.60% | 1.70% |
| Diagnosed Diabetes | Yes | 13.28% | 14.58% | 1.30% |
| Vegetable or Fruit Intake (1+ per Day) | No | 12.69% | 11.71% | 0.98% |
| Difficulty Walking | Yes | 15.85% | 16.78% | 0.94% |
| Leisure Physical Activity | No | 25.15% | 24.43% | 0.72% |
| Sex | Female | 56.57% | 55.87% | 0.69% |
| Smoked 100+ Cigarettes | Yes | 43.57% | 44.25% | 0.69% |
| General Health | Fair | 30.32% | 29.85% | 0.47% |
| Diagnosed Stroke | Yes | 3.66% | 4.05% | 0.40% |
| Heavy Alcohol Consumption | No | 94.19% | 94.39% | 0.20% |
| Age | missing_rate | 0.00% | 0.00% | 0.00% |

### Numeric Mean Shifts

| feature | train_mean | test_mean | mean_diff | standardized_diff | train_median | test_median |
| --- | --- | --- | --- | --- | --- | --- |
| Age | 54.74 | 57.72 | 2.98 | 0.17 | 56.00 | 59.00 |
| Body Mass Index | 28.19 | 28.40 | 0.21 | 0.03 | 27.07 | 27.30 |

### Observations

- The largest train/test differences are train-only missingness in high cholesterol and BMI.
- Test respondents are older on average than train respondents, so age-related validation checks are worth watching.
- Most non-missing categorical distributions are close enough for standard stratified CV, but missingness artifacts should be handled carefully.

## Relationship Patterns Summary

### Key Findings

1. **Clinical risk count is a compact, monotonic relationship feature.**
2. **General health, mobility, and stroke create high-signal interactions.**
3. **Train-only missingness is the largest drift source.**

### Correlations Identified

- Strong positive: general health rank, difficulty walking, age, diagnosed stroke, diabetes.
- Moderate positive: high blood pressure, high cholesterol, smoking, male sex.
- Negative: income rank, education rank, physical activity.

### Questions Raised

- Do risk-count and interaction features improve OOF PR-AUC/F1 in LightGBM/CatBoost?
- Should missingness flags be included, excluded, or regularized because test has no missing feature values?
