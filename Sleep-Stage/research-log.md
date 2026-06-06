# Research Log

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-06-06 15:51 +07 | bootstrap | Initialized autoresearch workspace for improving `sleep_stage_solution.ipynb`. Existing baseline is hand-crafted epoch features with sklearn HGB/ExtraTrees and fixed smoothing; best documented grouped weighted F1 is partial context HGB mean `0.51856`. Formed H1 raw-signal sequence model, H2 stronger tabular ensemble, and H3 calibrated sequence decoding. |
| 2 | 2026-06-06 16:05 +07 | bootstrap | Reviewed primary wearable sleep-staging literature: raw PPG/accelerometer CNNs, HR/actigraphy BLSTM, PPG transfer learning, InsightSleepNet, wav2sleep, and 2026 Mamba wearable staging. Synthesis supports raw-signal sequence modeling, context windows, class-imbalance handling, calibration, and ensembling. |
| 3 | 2026-06-06 16:12 +07 | inner-loop | H1 implementation scaffold: added `sleep_stage.deep` with raw epoch caches, per-recording normalization, centered context windows, Conv/Transformer model, mixed precision/TF32 H100 setup, grouped CV training, and final deep submission generation with transition decoding. Local tests pass; no H100 metric yet. |
| 4 | 2026-06-06 16:15 +07 | inner-loop | Updated `sleep_stage_solution.ipynb` so CUDA/H100 runs use the raw deep model by default. CPU/local runs keep the tabular pipeline fallback. Tabular CV is now optional via `RUN_TABULAR_CV=1`; deep CV is optional via `RUN_DEEP_CV=1`. |
