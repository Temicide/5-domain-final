# Zhang et al. 2017/2018: Multi-Level Feature Learning and RNNs Via Wearable Device

- URL: https://arxiv.org/abs/1711.00629
- DOI: https://doi.org/10.1016/j.compbiomed.2018.10.010
- Authors: Xin Zhang, Weixuan Kou, Eric I-Chao Chang, He Gao, Yubo Fan, Yan Xu
- Signals: Heart rate and wrist actigraphy from wearable devices.
- Method: Low/mid-level feature learning followed by BLSTM sequence classification.
- Reported result: Five-stage weighted F1 around 58% in both resting and comprehensive settings.
- Relevance: Directly supports sequence modeling for wearable sleep staging. The competition already shows a gain when adding previous/next context features.
- Actionable idea: Evaluate centered epoch windows and bidirectional temporal encoders with GroupKFold by full recording.
