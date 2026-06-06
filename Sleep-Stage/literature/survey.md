# Literature Survey: Wearable Sleep-Stage Classification

This survey tracks papers and resources relevant to improving the competition pipeline. The starting evidence supports three themes: raw wearable signals can support sleep staging, temporal sequence models are important, and validation must be subject/recording independent.

## Initial Search Targets

- Deep learning from wrist-worn PPG/accelerometry and other wearable signals.
- CNN/RNN/Transformer architectures for sleep staging.
- HRV, actigraphy, EDA, and temperature features for non-EEG sleep staging.
- Sequence decoding and sleep-stage transition priors.

## Current Working Synthesis

The current competition data is wearable-only: BVP, accelerometer, temperature, EDA, HR, and IBI at 16 Hz. This makes PSG-style EEG models such as SleepEEGNet or U-Time useful only as architectural inspiration. The more relevant direction is multimodal wearable modeling: per-epoch raw-signal encoders plus temporal context across epochs, with careful subject-independent validation.

## Papers Reviewed

| Paper | Signal fit to competition | Key takeaway |
| --- | --- | --- |
| Gillard et al., 2024, "Sleep staging classification from wearable signals using deep learning" | Very high: raw PPG plus 3-axis accelerometry | A deep CNN on raw wearable signals reached strong four-stage performance, supporting an end-to-end raw-signal model for this competition. |
| Zhang et al., 2017/2018, "Sleep Stage Classification Based on Multi-level Feature Learning and RNNs via Wearable Device" | High: heart rate plus wrist actigraphy | BLSTM sequence modeling improved wearable sleep staging and directly supports treating this as a sequence task. |
| Radha et al., 2021, "A deep transfer learning approach for wearable sleep stage classification with PPG" | High: PPG-derived wearable sleep staging | Transfer learning and recurrent modeling helped with limited PPG-labeled data, implying pretraining/regularization matter for 83-recording competitions. |
| Nam et al., 2024, "InsightSleepNet" | Medium-high: continuous PPG, not all competition channels | InceptionTime + TCN + local attention plus uncertainty estimation is a useful architecture pattern; class imbalance hurts deep sleep. |
| Carter and Tarassenko, 2024, "wav2sleep" | Medium-high: variable physiological inputs including PPG | Large multi-dataset physiological pretraining supports modality-robust encoders; too large for immediate competition reuse but validates raw sequence modeling. |
| Brach et al., 2026, "Mamba-based deep learning approach..." | Medium-high: wearable multimodal signals including PPG/accelerometry/temp | Modern sequence models can infer five sleep stages from non-EEG wearable data; ensembling variants helped. |

## Implications For This Repository

1. Replace the primary leaderboard candidate with a raw-signal neural model, not just more summary statistics.
2. Keep a strong tabular GBDT model as a calibration/blending baseline because the data size is modest.
3. Use temporal context explicitly: at least 5-11 epochs centered on the target epoch, and later full-recording decoding.
4. Track N3 and R separately; literature and local class imbalance both show minority-stage collapse is the main failure mode.
5. Add uncertainty/calibration outputs so Viterbi smoothing can use reliable probabilities instead of brittle argmax labels.
