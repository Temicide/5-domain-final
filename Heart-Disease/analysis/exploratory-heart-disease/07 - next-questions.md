
# Questions for Further Investigation

## Purpose

Exploratory analysis identified patterns that should be converted into concrete validation experiments before trusting Kaggle submission changes.

## Question 1: Do Clinical Interaction Features Improve OOF Score?

### What We Learned in Exploration

Clinical risk count is monotonic and high-signal.

### The Question

Do `clinical_risk_count`, `cardiometabolic_cluster`, `general_health_x_walking`, `age_x_high_bp`, and BMI/age bins improve 5-fold OOF ROC-AUC, PR-AUC, and F1-threshold proxy versus the current feature set?

### Why It Matters

These features are the most actionable route from EDA to leaderboard gain.

### Recommended Process

**Process Skill:** `hypothesis-testing` or a controlled modeling experiment.

### Data Required

- Current train/test CSVs.
- OOF predictions from baseline and feature-engineered models.

### Investigation Approach

1. Train baseline model with existing features.
2. Train same model/seed/folds with EDA features added.
3. Compare OOF ROC-AUC, PR-AUC, best F1, threshold, and test prediction distribution.

### Priority

**Priority:** High.

## Question 2: Should Missingness Flags Be Used?

### What We Learned in Exploration

Train has missing high cholesterol/BMI values; test does not.

### The Question

Do missingness flags improve real generalization, or do they overfit train-only artifacts?

### Why It Matters

This is the largest train/test drift source and can create misleading OOF gains.

### Recommended Process

**Process Skill:** `comparative-analysis`.

### Data Required

- Current train/test CSVs.
- Same fold split across variants.

### Investigation Approach

Compare three variants: no missing flags, missing category/imputation only, and explicit missing flags. Inspect OOF metrics and feature importance.

### Priority

**Priority:** High.

## Question 3: What Threshold Matches The Official Metric?

### What We Learned in Exploration

The target is imbalanced at 8.16%, and the submission requires labels.

### The Question

What OOF-tuned threshold should be used after verifying whether Kaggle scores F1, accuracy, balanced accuracy, or another label-based metric?

### Why It Matters

The same probability model can produce very different leaderboard scores depending on threshold.

### Recommended Process

**Process Skill:** `guided-investigation`.

### Data Required

- Official metric from Kaggle UI.
- OOF probabilities for top models.

### Investigation Approach

Tune threshold only on OOF predictions for the official metric proxy; do not tune threshold from public leaderboard feedback.

### Priority

**Priority:** High.

## Question 4: Which Categorical Strategy Wins?

### What We Learned in Exploration

Many important predictors are categorical strings, with some ordinal structure.

### The Question

Does native categorical handling, one-hot encoding, ordered encodings, or leakage-safe target encoding perform best across LightGBM/CatBoost/XGBoost?

### Why It Matters

Encoding choices can dominate tabular model performance.

### Recommended Process

**Process Skill:** `comparative-analysis`.

### Data Required

- Current train/test CSVs.
- Shared fold splits.

### Investigation Approach

Run model-family comparisons with identical folds and feature groups.

### Priority

**Priority:** Medium.

## Question 5: Can A Diverse Ensemble Improve PR-AUC/F1?

### What We Learned in Exploration

Signals include nonlinear clinical interactions and ordered socioeconomic gradients.

### The Question

Can LightGBM + CatBoost + logistic/diversity model OOF blending improve the official-metric proxy?

### Why It Matters

Ensembling often improves Kaggle tabular stability if base models make different errors.

### Recommended Process

**Process Skill:** `hypothesis-testing`.

### Data Required

- OOF/test predictions from candidate models.

### Investigation Approach

Blend OOF probabilities with constrained weights, tune threshold from blended OOF, and then apply the same blend/threshold to test.

### Priority

**Priority:** Medium.

## Questions Summary

### High Priority Questions

1. Do clinical interaction features improve OOF score?
2. Should missingness flags be used?
3. What threshold matches the official metric?

### Medium Priority Questions

1. Which categorical strategy wins?
2. Can a diverse ensemble improve PR-AUC/F1?

### Investigation Roadmap

1. Lock a stratified 5-fold split and baseline OOF metric table.
2. Run feature-engineering ablations from this EDA.
3. Run missingness handling comparison.
4. Verify official metric in Kaggle UI and tune threshold from OOF.
5. Blend top candidates only after individual gains are OOF-supported.

## Data Collection Priorities

Official metric verification is the highest-value missing external information. Survey year/geography would be useful but is not available in the competition files.
