# Nam et al. 2024: InsightSleepNet

- URL: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-024-02437-y
- DOI: https://doi.org/10.1186/s12911-024-02437-y
- Authors: Borum Nam, Beomjun Bark, Jeyeon Lee, In Young Kim
- Signals: Continuous PPG.
- Method: Local attention, InceptionTime, TCN, dense layers, and 1D CNN; energy-score uncertainty estimation.
- Reported result: On MESA, weighted F1 around 0.842 before energy thresholding, with improved selective performance after thresholding.
- Relevance: The architecture is well matched to raw BVP/PPG morphology and temporal context. The paper also notes deep sleep is hard under class imbalance.
- Actionable idea: Use Inception/TCN-style convolutional blocks or Transformer attention, add class weighting/focal loss, and save calibrated probabilities.
