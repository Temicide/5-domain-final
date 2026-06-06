# Sleep-Stage Competition Spec

## Objective

Build the highest-scoring solution for the Kaggle competition:

https://www.kaggle.com/competitions/super-ai-engineer-ss-6-individual-sleep-stage-classification

Primary metric: **weighted F1-score**.

The practical target is to beat the current accessible public leaderboard top score of about **0.62027 weighted F1**. Because the public leaderboard uses only 50% of the test data and the private leaderboard decides final placement, model selection must be driven by recording-group validation first and leaderboard feedback second.

## Kaggle Notebook Requirement

The deliverable must be a Kaggle Notebook ready solution. The notebook must:

- Run end-to-end in the Kaggle notebook environment.
- Read competition files from `/kaggle/input/...` when running on Kaggle, with local path fallbacks only for development.
- Generate `/kaggle/working/submission.csv`.
- Validate the generated CSV before saving or before declaring the run complete.
- Not submit through the Kaggle API, Kaggle CLI, or any other automated submission mechanism.

Manual upload or normal Kaggle notebook output use is allowed, but implementation must stop after generating `submission.csv`.

## Verified Competition Context

Source: authenticated Kaggle API via the local `kaggle` skill workflow.

- Competition opens: Friday, June 6, 2026, 10:15 Thailand time.
- Kaggle system closes: Saturday, June 6, 2026, 15:10 Thailand time.
- Format: individual competition.
- Submission limit: 5 submissions per day.
- Public leaderboard: 50% of test data, for development feedback.
- Private leaderboard: remaining 50% of test data, used for final ranking.
- Rules: no cheating, no hand-labeling or human prediction of validation/test records, follow Kaggle foundational rules.

## Data Profile

Local data root:

`Sleep-Stage/data/super-ai-engineer-ss-6-individual-sleep-stage-classification`

Files:

- `train/train/*.csv`: 83 continuous training recordings.
- `test_segment/test_segment/test001` through `test010`: 10 test subject folders.
- `sample_submission.csv`: 7,832 rows with `id,labels`.

Signal columns:

- `BVP`
- `ACC_X`
- `ACC_Y`
- `ACC_Z`
- `TEMP`
- `EDA`
- `HR`
- `IBI`
- `Sleep_Stage` in train only

Kaggle data description says all signals were resampled to **16 Hz**, and each **30-second segment** has one label. Therefore each scored epoch is:

`16 samples/sec * 30 sec = 480 rows`

Training scale:

- 83 train recordings.
- 32,037,600 train rows.
- 66,745 labeled 30-second train epochs.
- Per-recording length ranges from 345,600 to 458,400 rows.

Test scale:

- 10 test subjects.
- 7,832 test epochs.
- Every test CSV has exactly 480 rows.
- Segment IDs in sample submission map directly to test CSV names, e.g. `test001_00000`.

Training label distribution by epoch:

| Stage | Epochs | Percent |
| --- | ---: | ---: |
| N2 | 33,786 | 50.62% |
| W | 15,828 | 23.71% |
| N1 | 7,753 | 11.62% |
| R | 7,033 | 10.54% |
| N3 | 2,345 | 3.51% |

The dataset is heavily imbalanced. Weighted F1 rewards the majority stages most, but a model that collapses N3 or R into N2 will still lose meaningful score.

## Important Structure

Labels are constant over 480-row epochs and change sparsely across time. Common train transitions include:

- N1 -> N2
- W -> N1
- N2 -> N1
- N2 -> W
- N1 -> W
- N2 -> N3
- N3 -> N2
- N2 -> R

This makes sleep-stage prediction a sequence problem, not independent row classification and not even fully independent epoch classification. Best scoring should come from:

1. Strong epoch features.
2. Subject/recording-level normalization.
3. Neighboring-epoch context.
4. Temporal smoothing or sequence decoding.

## Baseline Findings

A quick segment-level feature cache was built in `/tmp/sleep_stage_segment_features_v1.npz` for research only. It used 82 per-epoch features:

- Mean, std, min, max, median, quartiles, mean absolute difference, and slope per signal.
- Accelerometer magnitude features.
- Coarse BVP FFT band-power ratios.

Validation used `GroupKFold` by training recording, never random row or random epoch split.

| Model | Grouped weighted F1 mean | Std | Notes |
| --- | ---: | ---: | --- |
| Most frequent class | 0.34142 | 0.04698 | Predicts N2 only. |
| Balanced logistic regression | 0.29334 | 0.03949 | Too linear; underfits. |
| ExtraTrees, balanced | 0.43452 | 0.02425 | Solid quick baseline. |
| HistGradientBoosting | 0.47296 | 0.02076 | Best completed simple-feature baseline. |
| HistGradientBoosting + prev/current/next context | 0.51856 | 0.02325 | Partial 3-fold run only; stopped to avoid a long job. |

One partial follow-up added previous/current/next epoch features, local differences, and normalized position within the recording. It completed 3 of 5 folds with scores `0.50196`, `0.55152`, and `0.50221`, for a partial mean of `0.51856`. The full run was stopped to avoid leaving a long CPU job active, but the partial result strongly supports prioritizing temporal context.

## Research Context

Wearable sleep-stage literature supports the same strategy:

- Wrist-worn accelerometer and PPG signals contain useful sleep-stage information through movement, heart-rate, and vascular dynamics. Google Research reported deep learning from raw wearable signals can approach human-like four-stage sleep scoring in favorable settings.
- Zhang et al. proposed wearable sleep-stage classification from heart rate and wrist actigraphy using low/mid-level features plus BLSTM sequence modeling, reporting the value of temporal dependencies.
- Radha et al. studied transfer learning for PPG-based wearable sleep-stage classification, supporting PPG-derived representations as useful when EEG is absent.
- Wearable sleep staging remains harder than PSG/EEG staging, so robust validation and temporal priors matter more than blindly fitting a high-capacity model.

Useful references:

- https://research.google/pubs/sleep-staging-classification-from-wearable-signals-using-deep-learning/
- https://www.sciencedirect.com/science/article/pii/S0010482518303032
- https://arxiv.org/abs/1711.00629
- https://www.nature.com/articles/s41746-021-00510-8
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7956647/

## Validation Protocol

Use this protocol for every serious experiment:

1. Convert train recordings into 30-second epochs of 480 rows.
2. Build features or model inputs per epoch.
3. Split by full recording, not by rows and not by shuffled epochs.
4. Use 5-fold `GroupKFold` over the 83 training recordings.
5. Report weighted F1 per fold, mean, and std.
6. Also report per-class F1 for W, N1, N2, N3, R.
7. Track confusion matrix, especially:
   - N1 vs W
   - N1 vs N2
   - N3 vs N2
   - R vs N1/N2
8. Only generate leaderboard-candidate CSVs for models that improve grouped validation or test a clearly isolated leaderboard hypothesis.

Never use row-level random splits. They leak adjacent samples from the same labeled segment and will overstate performance.

## Highest-Value Modeling Plan

### Phase 1: Strong Tabular Baseline

Build a reproducible `src/` pipeline:

- Parse train CSVs into epoch records.
- Parse test segment CSVs by `id`.
- Cache features as compressed NumPy/Parquet.
- Train and validate grouped models.
- Generate Kaggle `submission.csv`.

Feature set:

- Per-channel time statistics: mean, std, min, max, median, quantiles, IQR, skew, kurtosis.
- Change/motion statistics: mean absolute difference, max absolute difference, zero-crossing proxy.
- ACC magnitude and jerk features.
- BVP frequency bands and dominant frequency.
- HR/IBI features: mean, std, RMSSD-like successive difference, range, trend.
- EDA and TEMP slow trends.
- Cross-signal features: HR/IBI consistency, ACC magnitude x HR, EDA x TEMP.
- Subject-normalized versions: within-recording z-score/rank normalization where possible.
- Relative position in night: `epoch_index / n_epochs`.

Models:

- HistGradientBoosting as the first strong baseline.
- ExtraTrees or RandomForest as a diversity model.
- If installable, add LightGBM/CatBoost; these are likely stronger than sklearn HGB for this tabular problem.

Expected local grouped F1 target:

- Static features: `0.47-0.52`.
- With richer features and tuning: `0.52-0.56`.

### Phase 2: Sequence Context Without Deep Learning

This is the most urgent next improvement.

For each epoch, concatenate:

- Current epoch features.
- Previous 1-3 epoch features.
- Next 1-3 epoch features.
- Rolling mean/std over 3, 5, 9, and 15 epochs.
- Difference from previous and next epoch.
- Relative position in subject/night.

Train boosted trees on the expanded context table.

Expected gain:

- The partial HGB context test improved the first three comparable folds from the simple-feature HGB range into the low-to-mid `0.50s`, so a realistic full-CV target is `0.52-0.57`.

### Phase 3: Temporal Post-Processing

Use model probabilities, not hard labels.

Apply one or more of:

- Median/mode filter over 3-5 epochs to remove one-epoch spikes.
- Class-specific threshold adjustment for N3 and R.
- Transition-penalty Viterbi decoding using train transition counts.
- Prior constraints by night position:
  - More W at start/end.
  - N3 more likely in earlier/middle sleep cycles.
  - REM more likely later, but do not hard-code this too strongly.

Validation must compare raw vs smoothed predictions fold-by-fold. Do not tune smoothing only to public leaderboard.

Expected gain:

- `+0.01` to `+0.04` if probabilities are calibrated.

### Phase 4: Lightweight Deep Sequence Model

Only after the tabular/context baseline is reproducible.

Inputs:

- 480 x 8 raw epoch signal.
- Optional per-epoch feature vector.
- Sequence windows of 9-31 epochs around target epoch.

Candidate models:

- 1D CNN per epoch + BiLSTM/GRU over epochs.
- Temporal convolution network over epoch embeddings.
- Small transformer over epoch embeddings if compute allows.

Training:

- Grouped validation by recording.
- Class-weighted cross-entropy or focal loss.
- Mixup/noise augmentation on BVP/ACC only if validation improves.
- Early stopping on weighted F1, not loss.

This may exceed the available contest time. Use it only if Phase 1-3 plateau below public leaderboard.

### Phase 5: Ensemble and Submission Selection

Blend probability outputs from:

- HGB/static features.
- HGB/context features.
- ExtraTrees/context features.
- Optional LightGBM/CatBoost.
- Optional neural sequence model.

Use validation to choose blend weights. Start with simple weighted average:

- 50% best context boosting model.
- 25% static boosting model.
- 25% ExtraTrees or another decorrelated model.

Then apply temporal smoothing/Viterbi to the blended probabilities.

## Submission Strategy

With 5 submissions/day:

1. Generate a valid baseline CSV early to confirm format.
2. Generate a CSV from the best static feature model.
3. Generate a CSV from the best context feature model.
4. Generate a CSV from context model plus temporal smoothing.
5. Generate a CSV from the ensemble/smoothing variant.

Any actual leaderboard upload must be manual and outside the notebook. The notebook must not call Kaggle API/CLI submission commands.

Keep a submission log:

| Submission | Local CV | Public LB | Change |
| --- | ---: | ---: | --- |
| baseline | | | |
| static HGB | | | |
| context HGB | | | |
| smoothed context | | | |
| ensemble | | | |

Do not overfit to the public leaderboard. If a change improves public LB but hurts grouped CV, require a second related submission or a clear reason before trusting it.

## Implementation Tasks

1. Create a clean feature-generation script in `Sleep-Stage/src`.
2. Cache train/test epoch features with deterministic feature names.
3. Implement grouped CV evaluation with weighted F1 and per-class F1.
4. Implement valid submission generation from test segment IDs.
5. Train static HGB and ExtraTrees baselines.
6. Add context/rolling features and repeat grouped CV.
7. Add probability smoothing/Viterbi decoding.
8. Generate `submission.csv` from the best validated model.
9. If time remains, add LightGBM/CatBoost or a small CNN/GRU sequence model.

## Success Criteria

Minimum:

- Valid submission file for all 7,832 test IDs.
- Local grouped weighted F1 above `0.50`.
- Public leaderboard score above the simple baseline tier.

Competitive:

- Local grouped weighted F1 above `0.55`.
- Public leaderboard score near or above `0.62`.
- No large mismatch between grouped CV and public leaderboard behavior.

Winning attempt:

- Context boosted model plus temporal decoding and ensemble.
- Public leaderboard above current top while retaining grouped-CV robustness.
- Final selected submission is not chosen solely by public score.

## Key Risks

- Public leaderboard overfitting because only 50% of test is visible.
- Leakage from random epoch or row splits.
- Subject/domain shift: test has only 10 subjects, so individual physiology can dominate.
- N3 underprediction due to class imbalance.
- Sequence smoothing can improve common stages while damaging short true N1/R events.
- Sample submission has only the first three labels filled; all labels must be generated.

## Current Recommendation

The best path is **not** to start with a large neural model. The fastest high-scoring route is:

1. Rich 30-second epoch features.
2. Grouped validation by full recording.
3. Gradient-boosted tabular model.
4. Neighboring-epoch and rolling context features.
5. Probability smoothing or Viterbi decoding.
6. Small probability ensemble.

This matches both the local baseline evidence and the wearable sleep-stage literature.
