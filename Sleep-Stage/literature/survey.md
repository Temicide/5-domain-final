# Literature Survey: Wearable Sleep-Stage Classification

This survey tracks papers and resources relevant to improving the competition pipeline. The starting evidence supports three themes: raw wearable signals can support sleep staging, temporal sequence models are important, and validation must be subject/recording independent.

## Initial Search Targets

- Deep learning from wrist-worn PPG/accelerometry and other wearable signals.
- CNN/RNN/Transformer architectures for sleep staging.
- HRV, actigraphy, EDA, and temperature features for non-EEG sleep staging.
- Sequence decoding and sleep-stage transition priors.

## Current Working Synthesis

The current competition data is wearable-only: BVP, accelerometer, temperature, EDA, HR, and IBI at 16 Hz. This makes PSG-style EEG models such as SleepEEGNet or U-Time useful only as architectural inspiration. The more relevant direction is multimodal wearable modeling: per-epoch raw-signal encoders plus temporal context across epochs, with careful subject-independent validation.
