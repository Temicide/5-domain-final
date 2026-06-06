# Chest Disease Detection Competition Spec

Competition: https://www.kaggle.com/competitions/chest-disease-detection/data

Last updated: 2026-06-06

## Objective

Build the strongest possible Google Colab-ready A100 notebook for multi-label
chest X-ray disease detection. The notebook must train or fine-tune a model on
the competition train set, predict all required labels for every test image, and
write a valid `/content/submission.csv` for manual Kaggle upload.

This is a multi-label image classification task, not a single-label task. A
single radiograph may have multiple positive disease labels, while `No Finding`
is mutually exclusive with disease findings in the provided train data.

## Colab Notebook Requirement

The deliverable must be a Google Colab-ready notebook solution. The notebook
must:

- Run end-to-end in Google Colab from a fresh runtime.
- Target an A100 GPU runtime and use mixed precision by default.
- Install or import required packages in notebook cells.
- Authenticate to Kaggle inside Colab using either an uploaded `kaggle.json`
  file or Colab secrets/environment variables.
- Never print Kaggle credentials, file contents from `kaggle.json`, or secret
  values.
- Use the Kaggle CLI or Kaggle API inside Colab to download the competition
  data before reading any competition CSVs or images.
- Extract the downloaded competition archive into a Colab input directory such
  as `/content/input/chest-disease-detection`.
- Read competition files from `/content/input/...` after extraction when running
  in Colab.
- Use `/content/working` for caches, checkpoints, OOF predictions, and logs.
- Generate `/content/submission.csv`.
- Validate the generated CSV before declaring the run complete.
- Keep local development fallbacks under
  `/Users/temicide/Documents/5_domain_final/Chest-Disease`.
- Not submit through the Kaggle API, Kaggle CLI, or any other automated
  submission mechanism.

Manual upload of `/content/submission.csv` to Kaggle is allowed, but the
implementation must stop after creating and validating the CSV.

## Kaggle Data Access In Colab

The notebook must support both credential flows below without exposing secrets:

1. Uploaded `kaggle.json`:
   - Prompt the user to upload `kaggle.json` with `google.colab.files.upload()`.
   - Save it to `/root/.kaggle/kaggle.json`.
   - Set file permissions to `600`.
   - Do not display the uploaded JSON content.

2. Colab secrets or environment variables:
   - Read `KAGGLE_USERNAME` and `KAGGLE_KEY` from Colab secrets or environment
     variables.
   - Export them only into the current process environment.
   - Do not print either value.

After credentials are available, data download must run before CSV or image
loading, for example:

```bash
kaggle competitions download -c chest-disease-detection -p /content/input
```

Then extract the archive to:

```text
/content/input/chest-disease-detection
```

The local cached directory is named
`individual-test-chest-disease-detection`. The notebook should prefer the
downloaded Colab directory after extraction, but keep a local fallback for
development.

## Kaggle Context

The unauthenticated Kaggle web page may not expose all evaluation/rules
metadata. Before the final manual upload, verify the official metric and file
schema in the Kaggle UI.

Until the metric is confirmed, optimize in a metric-safe way:

- Primary ranking metric: mean ROC-AUC over disease labels.
- Primary threshold metric: macro F1 over all label columns.
- Secondary metrics: per-class F1, macro average precision, micro F1, and
  label-wise AUROC.
- Submission default: binary `0`/`1` predictions if the sample expects labels.
- Probability artifacts: save raw probabilities under `/content/working` so
  thresholds can be retuned without rerunning inference.

If Kaggle uses a probability ranking metric, submit calibrated probabilities. If
Kaggle uses F1 or accuracy-like metrics, submit thresholded binary predictions.
Do not tune thresholds only from public leaderboard feedback.

## Data Locations

Colab paths:

- Input root: `/content/input`
- Competition data directory: `/content/input/chest-disease-detection`
- Alternate extracted directory to detect:
  `/content/input/individual-test-chest-disease-detection`
- Working directory: `/content/working`
- Required submission path: `/content/submission.csv`
- Optional copied artifact path: `/content/working/submission.csv`

Local development fallbacks:

- Project root:
  `/Users/temicide/Documents/5_domain_final/Chest-Disease`
- Local data directory:
  `/Users/temicide/Documents/5_domain_final/Chest-Disease/data/individual-test-chest-disease-detection`
- Local image directory:
  `/Users/temicide/Documents/5_domain_final/Chest-Disease/data/individual-test-chest-disease-detection/images/images`
- Local output directory:
  `/Users/temicide/Documents/5_domain_final/Chest-Disease/outputs`
- Local submission fallback:
  `/Users/temicide/Documents/5_domain_final/Chest-Disease/outputs/submissions/submission.csv`

Path resolution should prefer Colab data only after the Kaggle download and
extraction have produced the expected CSV and image files. If Colab paths are
unavailable, use the local development fallback.

## Local Data Audit

Files:

- `data/individual-test-chest-disease-detection/train.csv`
- `data/individual-test-chest-disease-detection/test_submission.csv`
- `data/individual-test-chest-disease-detection/images/images/*.jpg`

Shapes and counts:

- Train: 9,963 rows, 14 columns.
- Test submission template: 2,506 rows, 14 columns.
- Images: 12,469 JPEG files.
- Every train and test filename resolves to a local image.
- Train filenames are unique.
- Test filenames are unique.
- The local data directory is about 19 GB.

Columns:

- ID column: `filename`.
- Label columns:
  - `Atelectasis`
  - `Cardiomegaly`
  - `Consolidation`
  - `Edema`
  - `Enlarged Cardiomediastinum`
  - `Fracture`
  - `Lung Lesion`
  - `Lung Opacity`
  - `No Finding`
  - `Pleural Effusion`
  - `Pleural Other`
  - `Pneumonia`
  - `Pneumothorax`

Important file quirks:

- The submission template is named `test_submission.csv`, not
  `sample_submission.csv`.
- In the local `test_submission.csv`, the first three rows contain non-empty
  example label values and the remaining 2,503 rows are blank. Treat this file
  as a schema and row-order template, not as ground truth.
- Generated submissions must fill every label cell for every test row.
- Preserve the exact test row order and exact `filename` values from
  `test_submission.csv`.

Representative image audit:

- Images are high-resolution grayscale-style JPEGs.
- A 120-image sample showed:
  - Width min/median/max: 1440 / 2524 / 3056 pixels.
  - Height min/median/max: 1616 / 2544 / 3056 pixels.
  - File size min/median/max: 0.80 / 1.57 / 2.85 MB.
  - Orientation counts: 82 portrait, 23 landscape, 15 square.

Implementation implication: do not assume fixed dimensions, square images, or a
single orientation. Use robust image loading, resize/pad transforms that preserve
aspect ratio when possible, and log corrupt/decode failures.

## Label Distribution

Training label prevalence:

| Label | Positives | Percent |
|---|---:|---:|
| No Finding | 3,200 | 32.12% |
| Pleural Effusion | 2,126 | 21.34% |
| Lung Opacity | 1,890 | 18.97% |
| Atelectasis | 1,721 | 17.27% |
| Cardiomegaly | 1,462 | 14.67% |
| Pneumonia | 1,283 | 12.88% |
| Lung Lesion | 905 | 9.08% |
| Edema | 902 | 9.05% |
| Pneumothorax | 818 | 8.21% |
| Enlarged Cardiomediastinum | 505 | 5.07% |
| Fracture | 495 | 4.97% |
| Consolidation | 454 | 4.56% |
| Pleural Other | 352 | 3.53% |

Labels per train image:

| Positive labels per image | Images |
|---:|---:|
| 1 | 6,019 |
| 2 | 2,322 |
| 3 | 1,148 |
| 4 | 375 |
| 5 | 88 |
| 6 | 11 |

Disease-positive labels per train image, excluding `No Finding`:

| Disease labels per image | Images |
|---:|---:|
| 0 | 3,200 |
| 1 | 2,819 |
| 2 | 2,322 |
| 3 | 1,148 |
| 4 | 375 |
| 5 | 88 |
| 6 | 11 |

`No Finding` is exclusive in train: all 3,200 `No Finding` rows have no disease
label positives. Enforce this consistency during threshold tuning and
post-processing unless validation shows otherwise.

Common co-occurrences:

- `Atelectasis` + `Pleural Effusion`: 891.
- `Lung Opacity` + `Pneumonia`: 592.
- `Atelectasis` + `Lung Opacity`: 567.
- `Lung Opacity` + `Pleural Effusion`: 532.
- `Cardiomegaly` + `Pleural Effusion`: 490.
- `Edema` + `Pleural Effusion`: 378.
- `Cardiomegaly` + `Edema`: 350.
- `Lung Lesion` + `Lung Opacity`: 345.

These correlations should influence error analysis and optional label-correlation
post-processing, but they should not be hard-coded without validation support.

## Evidence From Pipeline Research

Chest X-ray multi-label classification literature and tooling support a staged
approach:

- CheXpert frames the task as multi-label chest radiograph classification with
  uncertainty-aware labels and AUC-based evaluation over selected observations.
  Reference: https://arxiv.org/abs/1901.07031
- TorchXRayVision provides radiograph-specific pretrained models and exposes
  labels that overlap strongly with this competition, including `Atelectasis`,
  `Cardiomegaly`, `Consolidation`, `Edema`, `Fracture`, `Lung Lesion`,
  `Lung Opacity`, `Pleural Effusion`, `Pneumonia`, `Pneumothorax`, and
  `Enlarged Cardiomediastinum`.
  Reference: https://github.com/mlmed/torchxrayvision
- TorchXRayVision documentation shows practical DenseNet and ResNet models
  pretrained on NIH, CheXpert, PadChest, MIMIC-CXR, RSNA, and combined CXR
  datasets.
  Reference: https://torchxrayvision.readthedocs.io/
- Comparative chest X-ray classification studies report DenseNet-121 and other
  transfer-learning CNNs as strong baselines for CXR multi-label tasks.
  Reference: https://www.nature.com/articles/s41598-019-42294-8
- Transformer and modern convolutional backbones such as Swin and ConvNeXt-style
  models are reasonable A100 candidates after a domain-pretrained baseline is
  established.
  Reference: https://arxiv.org/abs/2206.04246

Practical conclusion: start with radiograph-domain pretrained models, then use
the A100 to train higher-resolution timm backbones and ensembles. Do not begin
with a from-scratch model.

Before using external CXR-pretrained weights, confirm the competition rules allow
public pretrained models or external data-derived weights. If external weights
are disallowed, fall back to ImageNet-pretrained timm models or train only from
allowed sources.

## Research Hypotheses

Use these as the initial autoresearch-style experiment ladder. Each experiment
should save OOF probabilities, test probabilities, thresholds, and a short
analysis note.

### H1: Domain-Pretrained CXR Backbones Win Early

Prediction: a TorchXRayVision DenseNet or ResNet pretrained on combined CXR
datasets will beat an ImageNet-only baseline at the same resolution because the
labels and radiograph distribution overlap with the competition.

Test:

- Fine-tune `densenet121-res224-all` or `resnet50-res512-all` with a 13-output
  head.
- Train with BCEWithLogitsLoss and class-balanced sampling or positive weights
  as isolated variants.
- Compare against an ImageNet-pretrained `tf_efficientnetv2_s` or ConvNeXt-Tiny
  baseline.

### H2: High Resolution Matters For Small/Rare Findings

Prediction: moving from 224/384 to 512/768 improves rare and localized labels
such as `Fracture`, `Lung Lesion`, `Pneumothorax`, and `Pleural Other`, but may
require careful regularization.

Test:

- Train the same backbone at 384, 512, and 768.
- Track per-class F1 and AUROC, not only macro averages.
- Use gradient accumulation if batch size becomes small.

### H3: Thresholds Are As Important As Backbones

Prediction: per-label thresholds tuned on OOF predictions will beat a global 0.5
threshold, especially for rare labels.

Test:

- Optimize thresholds per label for macro F1.
- Compare global threshold, per-label thresholds, and constrained
  `No Finding` post-processing.
- Lock thresholds using OOF only, then apply to test probabilities.

### H4: Multi-Scale Test-Time Augmentation Improves Robustness

Prediction: averaging predictions across horizontal flip, center crop, padded
resize, and 512/768 scales improves public/private stability.

Test:

- Use TTA only after single-view validation is stable.
- Compare validation gains and runtime cost.
- Save both raw and TTA probabilities.

### H5: Ensemble Diversity Beats Single Best Model

Prediction: a small ensemble of domain-pretrained DenseNet/ResNet plus modern
ConvNeXt/EfficientNet/Swin models will outperform any single checkpoint.

Test:

- Average OOF/test probabilities from 3 to 5 diverse models.
- Optimize ensemble weights on OOF macro F1 or mean AUROC.
- Prefer simple convex weighted averaging over complex stacking unless OOF size
  supports it.

## Validation Protocol

Use this protocol for every serious experiment:

1. Download and extract competition data in Colab before loading CSVs/images.
2. Read `train.csv` and `test_submission.csv`.
3. Resolve every filename to an image path and fail early on missing files.
4. Use a multi-label stratified split when possible, such as
   `iterative-stratification` `MultilabelStratifiedKFold`.
5. If iterative stratification is unavailable, use deterministic stratification
   on a compact label-signature fallback and report that limitation.
6. Use 5 folds for final candidates; 3 folds is acceptable for fast A100
   screening.
7. Keep fixed seeds and save the fold assignment.
8. Track per-fold:
   - mean ROC-AUC
   - macro average precision
   - macro F1
   - micro F1
   - per-class F1
   - per-class AUROC
   - per-class threshold
9. Tune thresholds out-of-fold only.
10. Save OOF predictions for every serious model under `/content/working/oof/`.
11. Save test probabilities for every serious model under
    `/content/working/test_preds/`.
12. Compare every experiment against the current OOF baseline, not just public
    leaderboard feedback.

Do not use random image-level splits if a future metadata file exposes patient
IDs or studies. If patient/study IDs become available, split by patient/study to
avoid leakage.

## A100 Training Configuration

Use PyTorch with AMP:

- `torch.cuda.amp.autocast` or `torch.amp.autocast`.
- `GradScaler` for fp16, or bf16 if stable.
- Enable `torch.backends.cudnn.benchmark = True` after fixing input sizes.
- Use `num_workers=2` to `8` depending on Colab stability.
- Use `pin_memory=True` and persistent workers if RAM allows.
- Cache decoded/resized images only if disk/RAM budget allows; the raw local
  dataset is large.

Starting settings:

| Resolution | Batch size target on A100 | Use case |
|---:|---:|---|
| 224 | 64-128 | TorchXRayVision quick baseline |
| 384 | 32-64 | fast screening |
| 512 | 16-48 | main training resolution |
| 768 | 8-24 | high-resolution final candidates |
| 1024 | 2-8 | exploratory rare-label/high-res crops |

Use gradient accumulation to keep the effective batch size stable if the A100
runtime memory varies.

## Preprocessing Requirements

Default preprocessing:

- Load images with PIL/OpenCV in grayscale or RGB consistently.
- For ImageNet/timm models, convert grayscale to 3 channels.
- For TorchXRayVision, follow its expected preprocessing range and input shape.
- Preserve aspect ratio with resize plus pad when feasible.
- Avoid aggressive cropping unless validated.
- Normalize using the pretrained model's expected statistics.
- Log image decode failures and replace failed images only with a controlled
  fallback that is visible in validation logs.

Recommended variants:

1. `resize_pad`
   - Resize longest side to target, pad to square.
   - Good default for preserving anatomy and orientation.

2. `center_crop`
   - Resize shorter side to target, center crop.
   - Often strong for CXR but can remove edge findings.

3. `clahe_light`
   - Apply light contrast-limited adaptive histogram equalization.
   - Test only as a controlled variant; do not assume it helps pretrained
     models.

4. `multi_crop_tta`
   - Center plus mild corner crops at inference.
   - Use for final candidates only.

Avoid transforms that can destroy clinical signal:

- heavy random rotation
- vertical flip
- strong color jitter
- aggressive random erasing over lung fields
- hard binarization
- uncontrolled cropping of borders or apices

Horizontal flip is usually acceptable for classification, but validate it
because laterality can matter for some radiographic findings.

## Modeling Ladder

### Phase 1: Reproducible Colab Baseline

Build one clean baseline notebook:

- Configure Kaggle credentials securely.
- Download and extract the competition archive.
- Resolve Colab and local fallback paths.
- Read `train.csv` and `test_submission.csv`.
- Build a PyTorch Dataset and DataLoader.
- Train one 13-output model with BCEWithLogitsLoss.
- Use multi-label stratified validation.
- Tune per-label thresholds on OOF predictions.
- Write and validate `/content/submission.csv`.

Suggested first model:

- `torchxrayvision` DenseNet, `densenet121-res224-all`, with a new 13-label
  head, or label-mapped outputs if using frozen pretrained logits as features.

Expected output artifacts:

- `/content/working/folds.csv`
- `/content/working/oof/baseline_probs.csv`
- `/content/working/test_preds/baseline_probs.csv`
- `/content/working/thresholds/baseline_thresholds.json`
- `/content/submission.csv`

### Phase 2: Strong Single Models

Train and compare:

- TorchXRayVision DenseNet121 CXR-pretrained.
- TorchXRayVision ResNet50 512 CXR-pretrained.
- `timm` ConvNeXt-Base or ConvNeXt-Large, ImageNet pretrained.
- `timm` EfficientNetV2 or NFNet-style model if stable.
- Swin/MaxViT/ViT-style model if training time allows.

Losses to test:

- BCEWithLogitsLoss baseline.
- BCEWithLogitsLoss with per-class `pos_weight`.
- Asymmetric Loss for multi-label imbalance.
- Focal loss as a controlled rare-label experiment.

Schedulers:

- AdamW optimizer.
- Cosine decay with warmup.
- OneCycleLR as a quick-screen alternative.
- Early stopping on OOF macro F1 or mean AUROC.

Regularization:

- weight decay around `1e-5` to `1e-3`.
- dropout/stochastic depth only if the backbone supports it cleanly.
- moderate augmentations, not natural-image-heavy augmentation.

### Phase 3: High-Resolution And Rare-Label Focus

For top candidates:

- Increase resolution to 512 or 768.
- Oversample images with rare positives.
- Use balanced batch sampling so rare labels appear regularly.
- Track rare-label validation separately.
- Try two-stage training:
  1. train head and upper backbone at lower resolution.
  2. unfreeze all layers and fine-tune at higher resolution with a lower LR.

Rare labels to watch:

- `Pleural Other`
- `Consolidation`
- `Fracture`
- `Enlarged Cardiomediastinum`
- `Lung Lesion`

### Phase 4: Thresholding And Label Consistency

Use OOF probabilities to tune:

- One global threshold.
- Per-label thresholds.
- Per-label thresholds with `No Finding` consistency.
- Optional top-k or minimum/maximum positive-label constraints.

Suggested `No Finding` rule to test:

- If any disease probability exceeds its tuned threshold, set `No Finding = 0`.
- If no disease label exceeds threshold, set `No Finding = 1`.
- Also test a separate `No Finding` threshold because the model may learn normal
  appearance better than the complement rule.

Do not hard-code co-occurrence rules unless OOF validation improves.

### Phase 5: Ensembles

Build the final model from a small set of diverse validated candidates:

- Domain-pretrained CXR model.
- High-resolution ConvNeXt/EfficientNet model.
- Transformer or hybrid model if it validates well.
- Different preprocessing view if it adds OOF diversity.

Ensemble methods:

- Simple average probabilities.
- Weighted average optimized on OOF.
- Rank averaging if calibration differs strongly.
- Per-label ensemble weights only if OOF evidence is stable.

Save final ensemble probabilities and thresholds before writing the binary CSV.

## Submission Contract

The final submission must match `test_submission.csv` schema:

```csv
filename,Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,Pneumothorax
cxr00001.jpg,0,0,0,0,0,0,0,0,1,0,0,0,0
```

Validation checks:

1. Columns exactly match `test_submission.csv`.
2. Row count equals 2,506.
3. `filename` values and order exactly match `test_submission.csv`.
4. No missing values in any label column.
5. Labels are numeric.
6. If submitting binary labels, all label values are `0` or `1`.
7. If submitting probabilities, all label values are in `[0, 1]`.
8. No duplicate filenames.
9. Save to `/content/submission.csv`.
10. Do not call any Kaggle submit command.

## Experiment Tracking

For every serious experiment, record:

- experiment name
- git/notebook version if available
- data root
- fold assignment file
- backbone and pretrained weights
- image size
- preprocessing
- augmentations
- loss
- optimizer and scheduler
- batch size and gradient accumulation
- epochs
- best epoch per fold
- OOF metrics
- per-class metrics
- thresholds
- prediction artifact paths
- notes on failure modes

Minimum artifact layout:

```text
/content/working/
  folds.csv
  logs/
  checkpoints/
  oof/
  test_preds/
  thresholds/
  submissions/
```

## Failure Modes To Guard Against

- Reading local files in Colab before downloading Kaggle data.
- Printing Kaggle credentials.
- Treating the task as single-label softmax classification.
- Training on `test_submission.csv` example-filled rows.
- Assuming `No Finding` can be positive together with disease labels.
- Using a global 0.5 threshold without OOF threshold tuning.
- Reporting only aggregate metrics while rare labels collapse.
- Cropping away relevant anatomy.
- Overfitting public leaderboard with repeated threshold tweaks.
- Forgetting to preserve test row order.
- Accidentally submitting probabilities when Kaggle expects binary labels, or
  binary labels when Kaggle expects probabilities.

## First Notebook Implementation Checklist

- Install `kaggle`, `timm`, `torchxrayvision`, `iterative-stratification`, and
  standard imaging dependencies.
- Securely configure Kaggle credentials.
- Download and extract competition data.
- Resolve `train.csv`, `test_submission.csv`, and image root.
- Audit data shapes and label counts in notebook output.
- Build multi-label folds.
- Train one fast baseline.
- Produce OOF probabilities.
- Tune thresholds.
- Train or refit final model strategy.
- Predict test probabilities.
- Apply thresholds and `No Finding` consistency.
- Write `/content/submission.csv`.
- Run submission validation cell.
- Stop.
