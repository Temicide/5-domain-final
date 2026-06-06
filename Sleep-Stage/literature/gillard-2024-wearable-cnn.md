# Gillard et al. 2024: Sleep Staging From Wearable Signals Using Deep Learning

- URL: https://research.google/pubs/sleep-staging-classification-from-wearable-signals-using-deep-learning/
- Authors: Ryan Gillard, Logan Schneider, Conor Heneghan, Logan Niehaus
- Year: 2024
- Signals: Raw PPG and 3-axis accelerometer from wrist-worn devices.
- Method: Deep CNN trained on raw wearable signals; pretraining used finger PPG records from MESA.
- Reported result: Four-stage accuracy around 0.79 and kappa around 0.66 on a withheld test set.
- Relevance: Closest support for using a raw PPG/accelerometer neural model on the competition data. The current repo has BVP plus accelerometry and should exploit those raw waveforms.
- Actionable idea: Implement an H100 mixed-precision CNN/Transformer/TCN model over raw 30-second windows, with neighboring-epoch context and augmentation.
