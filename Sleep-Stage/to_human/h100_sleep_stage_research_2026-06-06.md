# H100 Sleep-Stage Research Progress

Date: 2026-06-06

## Bottom Line

The current notebook was CPU-style: hand-crafted epoch statistics, shallow sklearn models, and fixed smoothing. The highest-probability improvement is now implemented as an H100-ready raw-signal Conv/Transformer path that consumes centered windows of BVP, ACC, TEMP, EDA, HR, and IBI.

No deep-model score is claimed yet. The code is scaffolded and locally verified; the next real metric must come from an H100 grouped fold.

## Why This Direction

Primary literature supports raw wearable sequence modeling:

- Google Research showed raw wrist PPG plus accelerometry can produce strong sleep-stage hypnograms with a deep CNN.
- Zhang et al. showed wearable HR/actigraphy benefits from BLSTM sequence modeling.
- Radha et al. showed transfer/regularization matters when PPG labels are limited.
- InsightSleepNet combines InceptionTime, TCN, attention, and uncertainty estimation for PPG sleep staging.
- wav2sleep and a 2026 Mamba wearable paper support modern physiological sequence encoders and ensembling.

This matches the local baseline evidence: adding temporal context already moved the documented grouped F1 from about `0.47` to partial `0.51856`.

## Implemented

- Added `src/sleep_stage/deep.py`
  - raw epoch cache for train/test
  - per-recording signal normalization
  - centered context windows, default `11 x 30s`
  - Conv1d downsampling stem
  - Transformer encoder over raw-signal tokens
  - class-weighted focal loss
  - H100 runtime settings: TF32, AMP/bfloat16, optional `torch.compile`
  - grouped CV training and final submission generation
  - Viterbi decoding from train transition priors
- Updated `sleep_stage_solution.ipynb`
  - H100/CUDA runs use the raw deep model by default
  - CPU/local runs keep the tabular fallback
  - tabular CV is skipped unless `RUN_TABULAR_CV=1`
  - deep CV runs only when `RUN_DEEP_CV=1`
- Added tests for deep windowing, normalization, model forward shape, and class weights.

Verification: `pytest` passed with `21 passed, 1 warning`.

## Recommended H100 Run Order

First sanity fold:

```python
%env RUN_H100_DEEP=1
%env RUN_DEEP_CV=1
%env DEEP_EPOCHS=6
%env DEEP_BATCH_SIZE=768
```

If the first fold is stable, run full CV:

```python
%env RUN_H100_DEEP=1
%env RUN_DEEP_CV=1
%env DEEP_EPOCHS=18
%env DEEP_BATCH_SIZE=1024
```

If memory is still low on H100 80 GB, increase `DEEP_BATCH_SIZE` to `1536` or `2048`. If validation is unstable, reduce to `512`, raise dropout to `0.2`, or reduce context from `11` to `7`.

Final submission run after CV:

```python
%env RUN_H100_DEEP=1
%env RUN_DEEP_CV=0
%env DEEP_EPOCHS=18
%env DEEP_BATCH_SIZE=1024
```

The notebook still writes `/content/submission.csv` and does not submit to Kaggle automatically.

## Next Experiments

1. H1 one-fold sanity: verify loss decreases, no NaN, class predictions are not collapsed to N2.
2. H1 full 5-fold CV: compare against `0.51856` baseline and inspect N1/N3/R F1.
3. H3 decoding sweep: tune transition strength and temperature on OOF probabilities.
4. H2 fallback: LightGBM/CatBoost with richer features if the raw model overfits.
5. H1.1 broaden: test Mamba/SSM block if Conv/Transformer works but plateaus.
