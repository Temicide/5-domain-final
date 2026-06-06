# H1 Protocol: Raw-Signal Sleep Transformer

## Hypothesis

A raw-signal Conv/Transformer sequence model trained on full 30-second epochs plus neighboring epochs will beat the current context HGB baseline because it can learn morphology, cross-channel timing, and sleep-transition dynamics that summary statistics discard.

## Prediction

On 5-fold GroupKFold by recording, the model should exceed the current documented partial baseline of `0.51856` weighted F1. A strong first target is `>=0.56`, with improved R and N3 F1 compared with tabular HGB.

## Method

1. Build a dataset that returns windows of `context_epochs x 480 x 8` raw samples centered on each labeled epoch.
2. Normalize per recording and per channel using train-fold statistics only.
3. Train a mixed-precision PyTorch model on H100:
   - Conv1d stem over channels/time.
   - Transformer encoder or Conformer-style temporal blocks.
   - Center-epoch classifier for fold CV; optional auxiliary heads for neighboring epochs.
4. Use class-balanced or focal loss to protect N1/N3/R.
5. Track weighted F1, per-class F1, confusion, and out-of-fold probabilities.

## Confirmatory Criteria

This hypothesis is supported if grouped weighted F1 improves by at least `+0.03` over `0.51856` and minority-stage F1 does not degrade materially.

## Risks

- Only 83 recordings may overfit a large model.
- Test subjects may differ from training recordings; heavy augmentation and validation discipline are needed.
- Context leakage must be avoided across validation recordings.
