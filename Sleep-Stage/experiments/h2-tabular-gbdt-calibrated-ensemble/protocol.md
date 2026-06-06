# H2 Protocol: Strong Tabular GBDT Ensemble

## Hypothesis

A stronger calibrated tabular ensemble using richer spectral/HRV/recording-normalized features and LightGBM/CatBoost will improve over sklearn HGB while staying robust on grouped CV.

## Prediction

Compared with the current context HGB baseline, LightGBM/CatBoost with expanded context and calibration should improve grouped weighted F1 by `+0.02` to `+0.05`, giving a robust fallback if the neural model overfits.

## Method

1. Expand features with Welch band powers, HRV features, robust percentiles, signal quality proxies, and per-recording normalized values.
2. Add context lags/windows up to 15-31 epochs, selected by grouped CV.
3. Train LightGBM/CatBoost with class weights, early stopping, and fold-specific validation.
4. Calibrate probabilities using out-of-fold predictions.
5. Blend with ExtraTrees/HGB only if OOF correlation and fold metrics justify it.

## Confirmatory Criteria

Support requires a full 5-fold grouped weighted F1 above the current baseline and better calibration for downstream sequence decoding.

## Risks

- Large context feature matrices may overfit and inflate local CV variance.
- LightGBM/CatBoost install/runtime differences in Colab must be handled gracefully.
