# H3 Protocol: Calibrated Sequence Decoding

## Hypothesis

Combining calibrated model probabilities with sleep-stage transition priors and tunable Viterbi smoothing will improve final sequence consistency without erasing rare stages.

## Prediction

Tuning transition strength, emission temperature, and minimum-duration penalties on grouped OOF probabilities should improve weighted F1 by `+0.005` to `+0.02` compared with raw argmax predictions, while preserving N3/R recall.

## Method

1. Save out-of-fold probabilities for every candidate model.
2. Build transition matrices from training recordings with smoothing.
3. Sweep emission temperature, transition multiplier, and mode-filter window on validation recordings.
4. Report overall and per-class deltas before using decoding on test.

## Confirmatory Criteria

Support requires consistent improvement on grouped OOF predictions and no large drop in minority-stage F1.

## Risks

- Fixed smoothing can hide model uncertainty and erase short R/N1/N3 bouts.
- Public LB may reward or punish smoothing differently from local folds because test has only 10 subjects.
