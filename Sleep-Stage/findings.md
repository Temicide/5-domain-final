# Research Findings

## Research Question

How can the wearable-signal sleep-stage pipeline be improved for best grouped weighted F1 and Kaggle leaderboard accuracy while using an H100 80 GB Colab Pro runtime effectively?

## Current Understanding

The current notebook is a compact Colab wrapper around a CPU-oriented scikit-learn pipeline. It converts each 30-second epoch into summary statistics, trains shallow classifiers, then applies fixed Viterbi/mode smoothing. This is reproducible and validation-aware, but it discards raw signal morphology and underuses the available H100.

The strongest documented local direction is temporal context: static hand-crafted features reached roughly `0.47` grouped weighted F1, while a partial context run reached about `0.51856`. That implies the problem should be treated as sequence modeling over nights/recordings, not as independent epoch classification.

The literature points to the same conclusion. Raw PPG/accelerometer CNNs, wearable HR/actigraphy BLSTMs, PPG recurrent transfer learning, TCN/InceptionTime attention models, and newer Mamba-style wearable sequence models all indicate that the highest-value improvement is a raw-signal temporal model with calibration and class-imbalance handling.

## Key Results

- Existing baseline from project spec: HistGradientBoosting static features grouped weighted F1 mean `0.47296`.
- Existing partial improvement: HistGradientBoosting with previous/current/next context reached partial 3-fold grouped weighted F1 mean `0.51856`.
- Competition data is available locally under `data/super-ai-engineer-ss-6-individual-sleep-stage-classification`, enabling validation and artifact generation before Colab H100 runs.
- Implementation result: added an H100-ready PyTorch raw-signal path and updated the notebook to use it by default when CUDA is available. Local verification: `21 passed, 1 warning`.

## Patterns and Insights

- Temporal context is already empirically valuable.
- Label imbalance is severe, especially N3, so optimization should track per-class F1 and confusion, not only weighted F1.
- H100 value will come from raw-signal models, large batches, mixed precision, and fold-parallel or fast repeated experiments, not from sklearn tree training.
- Per-recording normalization is likely necessary because wearable signals vary strongly by subject/device contact; using each recording's unlabeled signal statistics is also available at test time.
- Transition decoding should use calibrated probabilities and tuned strength, because fixed smoothing can improve hypnogram plausibility but can also erase short rare-stage bouts.

## Lessons and Constraints

- Use `GroupKFold` by recording for model selection; row-level or shuffled epoch splits leak adjacent samples.
- Keep `/content/submission.csv` generation but never automate Kaggle submission.
- Public leaderboard is only 50% of test data and should not overrule grouped validation.
- The test set has only 10 subjects, so sequence decoding must avoid over-smoothing rare labels.
- Do not spend H100 time on CPU-bound tabular CV unless explicitly requested; the notebook now defaults to skipping tabular CV.
- The raw deep model has not yet been scored; treat it as an implementation scaffold until a real grouped fold or H100 run produces metrics.

## Open Questions

- Does a raw-signal Conv/Transformer model improve enough over context HGB on grouped CV to justify being the primary submission model?
- Which context length works best: single epoch, 5-epoch window, 11-epoch window, or full-night sequence model?
- Does a tabular LightGBM/CatBoost ensemble remain competitive and better calibrated than the neural model?
- What transition-prior strength improves grouped F1 without erasing N1/N3/R?
- Can a Mamba/SSM encoder outperform the first Conv/Transformer once the H100 baseline is measured?

## Optimization Trajectory

Baseline trajectory starts at partial context HGB grouped weighted F1 `0.51856`. The next target is an H100-ready run that reaches at least `0.56` grouped weighted F1 and improves minority-stage F1 without collapsing N3/R.

No new model score is recorded yet. The next confirmatory run should set `RUN_DEEP_CV=1` for at least one fold, then proceed to full 5-fold CV if the first fold is not obviously broken.
