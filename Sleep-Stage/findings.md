# Research Findings

## Research Question

How can the wearable-signal sleep-stage pipeline be improved for best grouped weighted F1 and Kaggle leaderboard accuracy while using an H100 80 GB Colab Pro runtime effectively?

## Current Understanding

The current notebook is a compact Colab wrapper around a CPU-oriented scikit-learn pipeline. It converts each 30-second epoch into summary statistics, trains shallow classifiers, then applies fixed Viterbi/mode smoothing. This is reproducible and validation-aware, but it discards raw signal morphology and underuses the available H100.

The strongest documented local direction is temporal context: static hand-crafted features reached roughly `0.47` grouped weighted F1, while a partial context run reached about `0.51856`. That implies the problem should be treated as sequence modeling over nights/recordings, not as independent epoch classification.

## Key Results

- Existing baseline from project spec: HistGradientBoosting static features grouped weighted F1 mean `0.47296`.
- Existing partial improvement: HistGradientBoosting with previous/current/next context reached partial 3-fold grouped weighted F1 mean `0.51856`.
- Competition data is available locally under `data/super-ai-engineer-ss-6-individual-sleep-stage-classification`, enabling validation and artifact generation before Colab H100 runs.

## Patterns and Insights

- Temporal context is already empirically valuable.
- Label imbalance is severe, especially N3, so optimization should track per-class F1 and confusion, not only weighted F1.
- H100 value will come from raw-signal models, large batches, mixed precision, and fold-parallel or fast repeated experiments, not from sklearn tree training.

## Lessons and Constraints

- Use `GroupKFold` by recording for model selection; row-level or shuffled epoch splits leak adjacent samples.
- Keep `/content/submission.csv` generation but never automate Kaggle submission.
- Public leaderboard is only 50% of test data and should not overrule grouped validation.
- The test set has only 10 subjects, so sequence decoding must avoid over-smoothing rare labels.

## Open Questions

- Does a raw-signal Conv/Transformer model improve enough over context HGB on grouped CV to justify being the primary submission model?
- Which context length works best: single epoch, 5-epoch window, 11-epoch window, or full-night sequence model?
- Does a tabular LightGBM/CatBoost ensemble remain competitive and better calibrated than the neural model?
- What transition-prior strength improves grouped F1 without erasing N1/N3/R?

## Optimization Trajectory

Baseline trajectory starts at partial context HGB grouped weighted F1 `0.51856`. The next target is an H100-ready run that reaches at least `0.56` grouped weighted F1 and improves minority-stage F1 without collapsing N3/R.
