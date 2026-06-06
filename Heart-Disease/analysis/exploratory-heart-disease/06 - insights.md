
# Insights

## Insight Criteria

For this exploration, an insight must be:
- **Actionable:** informs modeling, validation, feature engineering, or submission generation.
- **Surprising:** non-obvious or easy to mishandle.
- **Meaningful:** large enough to affect Kaggle score.

## Insight 1: Clinical Risk Count Is A High-Value Engineered Feature

### The Finding

Combining high blood pressure, high cholesterol, stroke, diabetes, and walking difficulty into a clinical risk count creates a strong monotonic gradient.

### Why It's Significant

This compact feature gives linear/diversity models a signal that trees can learn but may still benefit from explicitly. It also creates a stable feature family for ensembling.

### Supporting Evidence

Found in: `04-relationship-patterns.md`.

| clinical_risk_count | n | positive_rate | lift_vs_base | bar |
| --- | --- | --- | --- | --- |
| 0 | 91597 | 1.38% | 0.17x | # |
| 1 | 60285 | 5.29% | 0.65x | #### |
| 2 | 41244 | 13.88% | 1.70x | ########### |
| 3 | 19966 | 23.29% | 2.85x | ################### |
| 4 | 7202 | 36.10% | 4.42x | ############################# |
| 5 | 1096 | 58.49% | 7.17x | ############################################### |

### Caveats and Limitations

Clinical risk count is not causal proof and may duplicate information already learned by tree splits.

### Confidence Level

**Confidence:** High.

**Reasoning:** The pattern is monotonic, uses large groups, and aligns with multiple univariate signals.

### Recommended Action

- Add `clinical_risk_count`, `cardiometabolic_cluster`, `age_x_high_bp`, and `general_health_x_walking` features.
- Validate feature impact against OOF PR-AUC/F1, not public leaderboard only.

## Insight 2: Train-Only Missingness Can Help CV And Hurt Test Generalization

### The Finding

Train has missing values in important predictors, especially `Told High Cholesterol` and `Body Mass Index`; test has no missing predictors.

### Why It's Significant

Missingness may correlate with target in train, but the competition test distribution cannot use that same missingness signal. This can inflate CV if feature handling is too aggressive.

### Supporting Evidence

Found in: `01 - data-familiarization.md`, `04-relationship-patterns.md`, and `05 - anomaly-investigation.md`.

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

### Caveats and Limitations

Because labels are unavailable for test, we cannot know whether missingness reflects true source-population differences or only data-entry artifacts.

### Confidence Level

**Confidence:** High.

**Reasoning:** The distribution mismatch is directly observed and affects high-signal predictors.

### Recommended Action

- Compare model variants with categorical `Missing` levels, numeric imputation, and missingness flags.
- If missingness flags increase OOF but shift feature importance strongly toward missing indicators, treat leaderboard gains cautiously.

## Insight 3: Age Needs Nonlinear Treatment

### The Finding

Age-bin target rates rise with age rather than behaving as a simple flat demographic variable.

### Why It's Significant

Linear models need bins/splines or interactions; tree models should still receive age bins because they are stable and interpretable.

### Supporting Evidence

Found in: `02-temporal-patterns.md`.

| age_bin | n | positive_rate | lift_vs_base | bar |
| --- | --- | --- | --- | --- |
| <=34 | 35394 | 0.71% | 0.09x | # |
| 35-44 | 28492 | 1.79% | 0.22x | ### |
| 45-54 | 38617 | 4.43% | 0.54x | ######## |
| 55-64 | 50997 | 8.26% | 1.01x | ############## |
| 65+ | 67890 | 16.78% | 2.06x | ############################# |

### Confidence Level

**Confidence:** High.

**Reasoning:** The pattern is monotonic across large bins and domain-plausible.

### Recommended Action

- Add age bins and age interactions with high blood pressure, cholesterol, diabetes, and sex.
- Avoid using raw ID/order as a temporal proxy.

## Insight 4: Socioeconomic Variables Are Ordered Risk Signals

### The Finding

Income and education levels show target-rate gradients and negative correlation with heart-disease history when encoded as higher-is-better ranks.

### Why It's Significant

Treating income and education as purely nominal categories discards ordinal structure that may improve generalization.

### Supporting Evidence

Found in: `03-segmentation-patterns.md` and `04-relationship-patterns.md`.

| feature | corr_with_target |
| --- | --- |
| income_rank | -0.118 |
| education_rank | -0.077 |

### Confidence Level

**Confidence:** Medium.

**Reasoning:** Direction is consistent, but survey socioeconomic categories can interact with age, health care access, and missingness.

### Recommended Action

- Add ordered encodings for income and education alongside raw categorical values.
- Test income/education interactions with general health and health-care access.

## Insight 5: Threshold Tuning Is Central To Submission Quality

### The Finding

The supervised positive rate is only 8.16%. Many high-risk segments are still minority-positive, so default `0.5` thresholds can under-predict positives for F1-like metrics.

### Why It's Significant

If the official metric rewards F1/balanced accuracy, the label threshold matters as much as small model AUC gains. If the metric is accuracy, the threshold should be different again.

### Supporting Evidence

Found in: target balance and segmentation rates.

| metric | value |
|---|---:|
| labeled rows | 221,390 |
| positive rows | 18,068 |
| base positive rate | 8.16% |

### Confidence Level

**Confidence:** Medium.

**Reasoning:** The need for threshold tuning is clear, but the official competition metric must be verified in the Kaggle UI.

### Recommended Action

- Generate OOF probabilities, tune threshold on OOF only, and write labels using the selected threshold.
- Keep separate candidate thresholds for F1, balanced accuracy, and accuracy until the official metric is confirmed.

## Non-Insights

### Pattern: ID Deciles Are Broadly Stable

**Why not an insight:** useful sanity check, but it does not suggest using ID as a feature.

### Pattern: Lifestyle Features Have Signal

**Why not an insight:** actionability is weaker than clinical and socioeconomic features; include them, but they are not primary improvement levers.

## Insights Summary

### Highest Priority Insights

1. **Clinical risk count and interactions:** likely useful feature engineering for Kaggle score.
2. **Train-only missingness:** biggest data-quality risk for generalization.
3. **Threshold tuning:** required because target is imbalanced and submission labels are `Yes`/`No`.

### Cross-Cutting Themes

The dataset is an imbalanced, mixed categorical tabular problem where clinical history, self-reported health status, and ordered socioeconomic/demographic structure carry the strongest signal.

### Strategic Implications

Prioritize feature-engineered GBDTs and OOF thresholding over deep models or raw one-hot baselines. Treat missingness and public leaderboard feedback carefully.
