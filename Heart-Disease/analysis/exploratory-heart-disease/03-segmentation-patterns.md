
# Segmentation Pattern Exploration

## Objective

Discover how heart-disease target rate varies across clinical, demographic, socioeconomic, access, and lifestyle groups.

## Exploration Approach

1. Compute univariate target rates for every categorical feature.
2. Inspect ordered BMI and age segments.
3. Check high-rate cross-segments that may become interaction features.

## Analysis 1: Strongest Categorical Segments

### Rationale

Target-rate lift by category identifies features that should receive careful encoding and interaction treatment.

### Query

```sql
SELECT
  '<feature>' AS feature,
  '<level>' AS segment,
  COUNT(*) AS n,
  AVG("History of HeartDisease or Attack" = 'Yes') AS positive_rate
FROM train_labeled
GROUP BY feature, segment
HAVING COUNT(*) >= 50
ORDER BY positive_rate DESC;
```

### Results

| feature | segment | n | positive_rate | lift_vs_base |
| --- | --- | --- | --- | --- |
| Diagnosed Stroke | Yes | 7949 | 37.69% | 4.62x |
| General Health | Very Poor | 9885 | 31.99% | 3.92x |
| Difficulty Walking | Yes | 34700 | 21.79% | 2.67x |
| Diagnosed Diabetes | Yes | 29212 | 20.49% | 2.51x |
| General Health | Poor | 27010 | 19.03% | 2.33x |
| High Blood Pressure | Yes | 85638 | 15.71% | 1.92x |
| Income Level | ($10,000 to less than $15,000 | 10878 | 15.55% | 1.91x |
| Told High Cholesterol | Yes | 79460 | 15.35% | 1.88x |
| Education Level | Elementary | 4404 | 14.60% | 1.79x |
| Education Level | Some high school | 9522 | 13.11% | 1.61x |
| Education Level | Never attended school | 237 | 13.08% | 1.60x |
| Income Level | $15,000 to less than $20,000 | 15477 | 12.83% | 1.57x |
| Income Level | Less than $10,000 | 10315 | 12.48% | 1.53x |
| Leisure Physical Activity | No | 55478 | 11.81% | 1.45x |
| Smoked 100+ Cigarettes | Yes | 96309 | 11.62% | 1.42x |
| Income Level | $20,000 to less than $25,000 | 18864 | 11.55% | 1.41x |
| Sex | Male | 96115 | 10.55% | 1.29x |
| Income Level | $25,000 to less than $35,000 | 23416 | 10.15% | 1.24x |
| Education Level | High school graduate | 57491 | 9.88% | 1.21x |
| Doctor Visit Cost Barrier | Yes | 22237 | 9.80% | 1.20x |
| Vegetable or Fruit Intake (1+ per Day) | No | 28035 | 9.56% | 1.17x |
| Cholesterol Checked | Yes | 182273 | 9.46% | 1.16x |
| General Health | Fair | 67055 | 8.81% | 1.08x |
| Education Level | Some college or technical school | 61833 | 8.44% | 1.03x |
| Health Care Coverage | Yes | 205154 | 8.43% | 1.03x |

### Observations

- Prior stroke, very poor/poor general health, difficulty walking, diabetes, high blood pressure, high cholesterol, male sex, low income, and low education all show elevated target rates.
- The strongest single segment is stroke history, with lift far above the base rate.

## Analysis 2: BMI Segments

### Rationale

BMI is continuous but clinically nonlinear; bins help check whether a tree model should receive explicit BMI class features.

### Query

```sql
SELECT bmi_class, COUNT(*) AS n, AVG(target_yes) AS positive_rate
FROM derived_bmi_classes
GROUP BY bmi_class;
```

### Results

| bmi_class | n | positive_rate | lift_vs_base | bar |
| --- | --- | --- | --- | --- |
| underweight | 3342 | 7.90% | 0.97x | ################## |
| normal | 67389 | 6.03% | 0.74x | ############## |
| overweight | 75081 | 8.49% | 1.04x | #################### |
| obese_i | 38645 | 10.38% | 1.27x | ######################## |
| obese_ii | 15253 | 11.11% | 1.36x | ########################## |
| obese_iii | 9955 | 11.32% | 1.39x | ########################## |
| Missing | 11725 | 4.52% | 0.55x | ########### |

### Observations

- BMI is not monotonic in the same clean way as age.
- Missing BMI has a lower target rate in train, but this missingness does not appear in test.

## Analysis 3: High-Rate Cross-Segments

### Rationale

Kaggle tabular scores often improve when important univariate signals are combined into compact interaction/risk-count features.

### Query

```sql
SELECT
  '<feature_a>=<level_a> | <feature_b>=<level_b>' AS interaction,
  COUNT(*) AS n,
  AVG("History of HeartDisease or Attack" = 'Yes') AS positive_rate
FROM train_labeled
GROUP BY interaction
HAVING COUNT(*) >= 300
ORDER BY positive_rate DESC
LIMIT 20;
```

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
| Income Level=$50,000 to less than $75,000 | General Health=Very Poor | 788 | 31.98% | 3.92x |
| Diagnosed Stroke=Yes | General Health=Fair | 2454 | 31.38% | 3.84x |
| Income Level=$35,000 to less than $50,000 | General Health=Very Poor | 1051 | 30.83% | 3.78x |
| Income Level=Less than $10,000 | General Health=Very Poor | 1603 | 30.44% | 3.73x |
| Diagnosed Stroke=Yes | Difficulty Walking=No | 4035 | 28.70% | 3.52x |
| Diagnosed Stroke=No | General Health=Very Poor | 8274 | 27.28% | 3.34x |
| Sex=Male | Age=79 | 983 | 26.25% | 3.22x |
| Sex=Male | Age=76 | 961 | 26.22% | 3.21x |
| Sex=Male | Age=77 | 1003 | 26.22% | 3.21x |
| Sex=Male | Age=75 | 931 | 26.10% | 3.20x |

### Observations

- Stroke plus poor health/walking difficulty creates very high-risk regions.
- General health and walking difficulty interact strongly enough to justify explicit interaction features.
- Diabetes, blood pressure, cholesterol, and BMI should likely be grouped into a cardiometabolic feature family.

## Segmentation Patterns Summary

### Key Findings

1. Clinical history features dominate target-rate lift.
2. General health and mobility variables are among the strongest non-diagnosis survey signals.
3. Socioeconomic variables encode meaningful risk gradients and should be treated as ordered, not just nominal.

### Notable Differences

- Positive rates for high-risk clinical segments are several times the base rate.
- Male positive rate is higher than female positive rate, but sex is weaker than stroke/general health/walking difficulty.

### Questions Raised

- Which categorical encodings best preserve ordinal structure without leaking target statistics?
- Do explicit clinical-risk interactions improve OOF PR-AUC/F1 beyond native tree splits?
