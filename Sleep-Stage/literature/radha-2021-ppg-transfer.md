# Radha et al. 2021: Deep Transfer Learning for PPG Sleep Staging

- URL: https://research.tue.nl/en/publications/a-deep-transfer-learning-approach-for-wearable-sleep-stage-classi/
- DOI: https://doi.org/10.1038/s41746-021-00510-8
- Authors: Mustafa G. Radha, Pedro Fonseca, Arnaud Moreau, Marco Ross, Andreas Cerny, Peter Anderer, Xi Long, Ronald M. Aarts
- Signals: PPG sleep staging with transfer from ECG.
- Method: Deep recurrent model pretrained on ECG recordings and adapted to a smaller PPG dataset.
- Reported result: Best transfer strategy reached kappa about 0.65 and accuracy about 76% for four classes.
- Relevance: This competition has only 83 training recordings, so strong regularization, augmentation, and possibly external pretraining are plausible high-value directions.
- Actionable idea: Start without external data for competition simplicity, but design the raw model to accept pretrained weights later.
