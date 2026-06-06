# Carter and Tarassenko 2024: wav2sleep

- URL: https://arxiv.org/abs/2411.04644
- Authors: Jonathan F. Carter, Lionel Tarassenko
- Venue: ML4H 2024
- Signals: Variable physiological inputs including ECG, PPG, and respiratory signals.
- Method: Unified model trained across over 10,000 overnight recordings from six public PSG datasets.
- Relevance: Supports modality-robust physiological sequence pretraining. For this competition, the useful lesson is not to hand-engineer one signal path too narrowly.
- Actionable idea: Build the local model around channel-agnostic signal embeddings and optional channel dropout so it can learn robust multimodal features.
