# Sleep-Stage Competition Spec

## Objective

Build the highest-scoring solution for the Kaggle competition:

https://www.kaggle.com/competitions/super-ai-engineer-ss-6-individual-sleep-stage-classification

Primary metric: **weighted F1-score**.

The practical target is to beat the current accessible public leaderboard top score of about **0.62027 weighted F1**. Because the public leaderboard uses only 50% of the test data and the private leaderboard decides final placement, model selection must be driven by recording-group validation first and leaderboard feedback second.

## Colab Notebook Requirement

The deliverable must be a Google Colab-ready solution. The notebook must:

- Run end-to-end in Google Colab from a fresh runtime.
- Use the Kaggle CLI or Kaggle API inside Colab to download the competition data before any data files are read.
- Extract the downloaded archive in Colab before training, validation, or prediction begins.
- Handle Kaggle credentials securely by using either an uploaded `kaggle.json` file or Colab secrets/environment variables. The notebook must never print the username, key, or full credential file contents.
- Use Colab defaults for runtime paths:
  - `/content` as the runtime root.
  - `/content/input` as the extracted competition data root.
  - `/content/working` for caches and intermediate outputs.
  - `/content/submission.csv` as the final submission CSV.
- Retain local development fallbacks under `/Users/temicide/Documents/5_domain_final/Sleep-Stage`, including the existing local data directory and local `working/` output directory.
- Generate and validate `submission.csv` before declaring the run complete.
- Stop after creating the CSV. Do not automate competition submission through the Kaggle CLI, Kaggle API, notebook code, or any other mechanism.

Manual upload of the generated CSV to Kaggle is allowed outside the notebook flow, but the implementation must not contain or execute a competition submission command.

## Verified Competition Context

Source: authenticated Kaggle API via the local `kaggle` skill workflow.

- Competition opens: Friday, June 6, 2026, 10:15 Thailand time.
- Kaggle system closes: Saturday, June 6, 2026, 15:10 Thailand time.
- Format: individual competition.
- Submission limit: 5 submissions per day.
- Public leaderboard: 50% of test data, for development feedback.
- Private leaderboard: remaining 50% of test data, used for final ranking.
- Rules: no cheating, no hand-labeling or human prediction of validation/test records, follow Kaggle foundational rules.

## Data Access And Paths

Competition slug:

`super-ai-engineer-ss-6-individual-sleep-stage-classification`

Colab data flow:

1. Start from `/content`.
2. Create `/content/input` and `/content/working`.
3. Configure Kaggle credentials without printing secrets.
4. Download the competition archive using one of:
   - `kaggle competitions download -c super-ai-engineer-ss-6-individual-sleep-stage-classification -p /content/input`
   - `KaggleApi().competition_download_files(..., path="/content/input")`
5. Extract the archive into `/content/input/super-ai-engineer-ss-6-individual-sleep-stage-classification`.
6. Read train, test, and sample submission files only after extraction has completed.
7. Write the final CSV to `/content/submission.csv`.

Local development fallback:

`/Users/temicide/Documents/5_domain_final/Sleep-Stage/data/super-ai-engineer-ss-6-individual-sleep-stage-classification`

Local output fallback:

`/Users/temicide/Documents/5_domain_final/Sleep-Stage/working/submission.csv`

## Data Profile

Files after extraction:

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

1. Download and extract the competition data in Colab, or resolve the local fallback data root during development.
2. Convert train recordings into 30-second epochs of 480 rows.
3. Build features or model inputs per epoch.
4. Split by full recording, not by rows and not by shuffled epochs.
5. Use 5-fold `GroupKFold` over the 83 training recordings.
6. Report weighted F1 per fold, mean, and std.
7. Also report per-class F1 for W, N1, N2, N3, R.
8. Track confusion matrix, especially:
   - N1 vs W
   - N1 vs N2
   - N3 vs N2
   - R vs N1/N2
9. Only generate leaderboard-candidate CSVs for models that improve grouped validation or test a clearly isolated leaderboard hypothesis.

Never use row-level random splits. They leak adjacent samples from the same labeled segment and will overstate performance.

## Highest-Value Modeling Plan

### Phase 1: Strong Tabular Baseline

Build a reproducible `src/` pipeline:

- Configure Colab paths and local fallbacks.
- Download and extract the competition archive before loading data.
- Parse train CSVs into epoch records.
- Parse test segment CSVs by `id`.
- Cache features as compressed NumPy/Parquet under `/content/working` in Colab or local `working/` during development.
- Train and validate grouped models.
- Generate and validate `submission.csv`.

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
- If installable in Colab without destabilizing the runtime, add LightGBM/CatBoost; these are likely stronger than sklearn HGB for this tabular problem.

Expected grouped F1 target:

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

## Submission Output Contract

The final run must produce:

- `/content/submission.csv` in Colab.
- `/Users/temicide/Documents/5_domain_final/Sleep-Stage/working/submission.csv` for local fallback runs.

The submission file must:

- Have exactly the columns `id,labels`.
- Match `sample_submission.csv` row count and ID order exactly.
- Use only labels `W`, `N1`, `N2`, `N3`, and `R`.
- Contain no missing labels.

The notebook must print validation summaries such as row count, ID match status, label counts, and the output path. It must not print Kaggle credentials and must not run any competition submission command.
