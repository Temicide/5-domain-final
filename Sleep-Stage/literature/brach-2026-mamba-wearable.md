# Brach et al. 2026: Mamba-Based Wearable Sleep Staging Without EEG

- URL: https://academic.oup.com/sleep/advance-article-abstract/doi/10.1093/sleep/zsag022/8466336
- DOI: https://doi.org/10.1093/sleep/zsag022
- Signals: Wearable ECG, triaxial accelerometry, chest temperature, finger PPG, and finger temperature.
- Method: Mamba-based recurrent neural network with ensembling.
- Reported result: Five-class F1 about 66.15% on adults from a tertiary sleep clinic.
- Relevance: Shows modern long-context sequence models can work on non-EEG wearable sleep staging, including modalities close to the competition data.
- Actionable idea: Transformer is a practical first H100 model; Mamba/SSM blocks are a strong later experiment if installation is stable in Colab.
