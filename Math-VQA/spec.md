# Thai Math VQA Colab Pipeline Spec

Source: https://www.kaggle.com/competitions/super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen

## Objective

Build a Google Colab-ready notebook for a Kaggle competition that reads one image of a Thai or English mathematics problem and predicts the final answer as a short string.

The first implementation target is a zero-shot or few-shot VQA pipeline using LLaVA through Ollama:

https://ollama.com/library/llava

The notebook must download the competition data into Colab, then produce a valid `submission.csv` for manual Kaggle upload.
The notebook must not submit through the Kaggle API, Kaggle CLI, or any other automated submission mechanism. It must only use the Kaggle API or CLI for data download.

## Non-goals

- Do not train a new multimodal model from scratch.
- Do not manually label test images.
- Do not depend on public leaderboard tuning only. The competition split is adversarial, so public leaderboard behavior may not represent private leaderboard behavior.
- Do not build a general math solver outside the competition scope. The output only needs to be the final answer string.
- Do not call `kaggle competitions submit`, Kaggle API submission endpoints, or notebook-side submission automation.

## Dataset

Local dataset path:

`Math-VQA/data/super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen/`

Google Colab dataset path after download and extraction:

`/content/input/super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen/`

Google Colab runtime artifact path:

`/content/working/`

Required final Google Colab submission path:

`/content/submission.csv`

Expected Colab files after download:

```text
/content/input/super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen/sample_submission.csv
/content/input/super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen/train.csv
/content/input/super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen/test.csv
/content/input/super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen/images/images/623.jpg
```

| File                    | Local rows | Columns                      | Purpose                                                         |
| ----------------------- | ---------: | ---------------------------- | --------------------------------------------------------------- |
| `train.csv`             |        280 | `id`, `image_path`, `answer` | Labeled training examples.                                      |
| `test.csv`              |        420 | `id`, `image_path`           | Unlabeled examples to predict.                                  |
| `sample_submission.csv` |        420 | `id`, `answer`               | Submission template. Current sample predicts `2` for every row. |
| `images/images/*.jpg`   |        700 | n/a                          | One JPEG per problem id.                                        |

Verified local data facts:

- Train ids and test ids do not overlap.
- Every train and test `image_path` resolves to an existing local image.
- Train answers have 198 unique strings.
- Common train answers include `1`, `2`, `3`, `4`, `5`, `8`, `36`, `80 องศา`, `6`, and `45`.
- Answer formats include plain integers, decimals, numbers with Thai units, LaTeX expressions, Thai digits, and short Thai phrases.

## Data Composition

The competition page says the 700 problems come from 9 source buckets, roughly corresponding to grade level and source collection.

| Bucket | Count |
| ------ | ----: |
| 101    |    26 |
| 102    |    27 |
| 103    |   125 |
| 104    |   134 |
| 105    |   101 |
| 116    |   105 |
| 118    |    38 |
| 120    |    30 |
| 122    |   114 |

Important split warning:

- Train has a representative mix.
- Some source buckets are evaluated only on the public leaderboard.
- Some source buckets are evaluated only on the private leaderboard.
- Avoid choosing prompts, preprocessing, or postprocessing rules only because they improve one public leaderboard attempt.

## Google Colab Runtime

The notebook target environment is:

- Google Colab Linux runtime.
- Prefer a GPU runtime. T4 is acceptable; faster GPUs reduce local LLaVA latency.
- Colab working directory defaults to `/content`.

Implementation risks:

- Colab does not include the Kaggle CLI by default.
- Colab does not include the competition data by default; the notebook must download it.
- Kaggle competition downloads require credentials and accepted competition rules.
- Ollama may not be available by default in Colab notebooks.
- The Ollama Linux installer may require the `zstd` command to extract the `.tar.zst` package. A Colab runtime can fail with `ERROR: This version requires zstd for extraction. Please install zstd and try again`.
- Colab has internet when enabled, but package/model downloads can still be slow or interrupted.
- Pulling the LLaVA model during notebook execution may be too slow for the session.
- Individual LLaVA/Ollama image requests may exceed a fixed HTTP read timeout. A Colab runtime can fail with `ReadTimeout: HTTPConnectionPool(host='localhost', port=11434): Read timed out`.

Required mitigation:

- The notebook must install the Kaggle CLI with `pip install kaggle`.
- The notebook must configure Kaggle credentials from one of these sources:
  - Environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.
  - Existing `/root/.kaggle/kaggle.json`.
  - Uploaded `kaggle.json` through `google.colab.files.upload()` when running in Colab.
- The notebook must keep Kaggle credentials secure by writing `kaggle.json` only to `/root/.kaggle/kaggle.json`, setting file permissions to `600`, never printing credential values, and never committing or persisting credentials in notebook output artifacts.
- The notebook must download competition data using the Kaggle CLI or Kaggle API before loading any CSVs or images. A CLI implementation should use `kaggle competitions download`.
- The notebook must extract the downloaded archive into `/content/input/super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen/` before any reads from `train.csv`, `test.csv`, `sample_submission.csv`, or image files.
- If Kaggle credentials are missing, data download fails, or the user has not accepted competition rules, the notebook must fail with a readable message before data loading.
- The notebook must contain a clear setup cell for Ollama and LLaVA.
- Before running the Ollama install script, the setup cell must check whether `zstd` is on `PATH`. If missing and `apt-get` is available, it must install `zstd` with `apt-get update` and `apt-get install -y --no-install-recommends zstd`.
- If `zstd` cannot be installed, the notebook must fail before the Ollama installer with a readable error that says Ollama requires `zstd` and suggests using a runtime with `zstd`, packaging Ollama/model assets, or switching to the Hugging Face fallback.
- If Ollama setup fails, the notebook must fail with a readable error message.
- The notebook must expose configurable per-image request controls: `OLLAMA_REQUEST_TIMEOUT` defaulting to at least 600 seconds and `OLLAMA_RETRIES` defaulting to at least 1 retry.
- A timeout or request failure for one image must not abort the whole test inference loop. The notebook must log the error, use the training-prior fallback answer for that image, set `used_fallback=True`, and continue producing a valid submission.
- Prefer packaging required model assets as a Kaggle dataset if internet/model download is not reliable.
- Keep a documented fallback option to a Kaggle-compatible Hugging Face vision-language model if Ollama cannot run.

## Pipeline Contract

### Input

The notebook reads:

- `test.csv`
- `sample_submission.csv`
- JPEG images referenced by `test.csv`

The notebook may also read `train.csv` for:

- Prompt examples
- Local validation
- Answer prior fallback
- Postprocessing rule development

### Output

The notebook writes:

```text
/content/submission.csv
```

The notebook may also copy the same final CSV to `/content/working/submission.csv` for convenience, but `/content/submission.csv` is the required final artifact.

`submission.csv` requirements:

- Header must be exactly `id,answer`.
- Include every `id` from `test.csv` exactly once.
- Preserve row order from `sample_submission.csv`.
- Treat `id` as a string.
- Treat `answer` as a string.
- No empty answers.
- `answer` may contain Thai characters, LaTeX, units, or plain numbers.

## Submission Format

Example:

```csv
id,answer
1,2
10,2
100,2
```

Kaggle constraints:

- Submission limit: 5 submissions per participant.
- Final private scoring can use up to 2 selected submissions.
- If no submissions are selected, Kaggle uses the two highest public leaderboard submissions by default.

## Evaluation

Metric: accuracy after normalizing both prediction and ground truth.

The normalization steps listed on Kaggle are:

1. Lowercase and strip outer whitespace.
2. Convert Thai digits `๐ ๑ ๒ ๓ ๔ ๕ ๖ ๗ ๘ ๙` to Arabic digits `0-9`.
3. Remove dollar signs used as inline LaTeX delimiters.
4. Remove recognized Thai and English unit words, for example `ตารางเซนติเมตร`, `ลูกบาศก์หน่วย`, `เซนติเมตร`, `องศา`, `หน่วย`, `จำนวน`, `วิธี`, `แบบ`, `ค่า`, `ร้อยละ`, `ดอลลาร์`, `บาท`, `degrees`, `square centimeters`, and `years old`.
5. Expand selected LaTeX macros:
   - `\frac{a}{b}` -> `(a)/(b)`
   - `\sqrt{x}` -> `sqrt(x)`
   - `\pi` -> `pi`
   - `\times`, `\cdot` -> `*`
   - `\div` -> `/`
   - `\pm` -> `+-`
   - `\overrightarrow{AB}`, `\overline{AB}`, `\vec{AB}` -> `AB`
   - `\left`, `\right`, `\,`, `\;`, `\:`, `\!` -> removed
6. Remove whitespace and structural characters `{ } \ ,`.
7. Remove redundant parentheses around pure integers, for example `(3)` -> `3`.
8. Canonicalize integer-valued numbers, for example `2.0` -> `2`.

Examples:

| Raw answer          | Normalized |
| ------------------- | ---------- |
| `20 ตารางเซนติเมตร` | `20`       |
| `30 องศา`           | `30`       |
| `$6\sqrt{3}$`       | `6sqrt3`   |
| `$\frac{17}{10}$`   | `17/10`    |
| `๒๕`                | `25`       |
| `2.0`               | `2`        |

## EDA Findings

Observed image and answer characteristics:

- Problems mix Thai and English.
- Some questions contain diagrams, graphs, shapes, or visual choices, not only text.
- Some images contain small powers or superscripts on variables.
- Some images include graph axes and geometric labels.
- Images 82 and 197 contain unusual symbols.
- Images 94, 101, 134, 140, 162, and 200 are low resolution.
- Image 95 appears not to be a normal mathematics problem.
- Image 156 has high black point or poor contrast.
- Image 451 is shape-heavy and may require visual reasoning.
- Image 569 requires context: the answer is not a plain printed number, but a number formed by straight lines or sticks.
- Image dimensions vary significantly. Some images are very wide and short, so preprocessing must preserve aspect ratio and avoid losing edge context.

## Preprocessing Requirements

Preprocessing must be conservative. Many examples depend on exact visual layout, small symbols, diagrams, and graph details.

Required default preprocessing:

- Load every image in RGB.
- Preserve aspect ratio.
- Do not crop by default.
- Do not globally binarize by default.
- Do not remove borders or whitespace unless the transformation is validated.
- Keep enough resolution for small exponents, graph labels, geometry labels, and Thai characters.
- Record the final image size passed to the VQA model.

Recommended preprocessing variants to test:

1. **Raw image**
   - RGB image with only model-required resizing.
   - This is the baseline.

2. **Upscaled image**
   - Use for low-resolution images.
   - Preserve aspect ratio.
   - Useful candidates from EDA: 94, 101, 134, 140, 162, 200.

3. **Contrast-enhanced image**
   - Use for high black point or low-contrast images.
   - Useful candidate from EDA: 156.

4. **High-resolution pass**
   - Use for shape-heavy, graph-heavy, or superscript-heavy cases.
   - Useful candidates from EDA: 451, 569.

Preprocessing must not:

- Destroy thin lines in geometry or graph problems.
- Remove visual answer choices.
- Convert diagrams into unreadable blobs.
- Crop out axes, labels, answer choices, or edge text.

## Prompting Requirements

The model should return only the final answer. Explanations make postprocessing harder and may create invalid submissions.

Base prompt:

```text
You are solving a Thai/English math problem from an image.
Read all text, diagrams, graphs, shapes, and answer choices carefully.
Return only the final answer.
Do not explain.
Do not include "answer:".
Use Arabic digits when possible.
```

Variant prompt for diagram-heavy images:

```text
You are solving a math problem from an image.
The image may contain Thai text, English text, diagrams, graphs, shapes, or visual answer choices.
Use the visual information, not only OCR text.
Return only the final answer.
```

Variant prompt for strict formatting:

```text
Return exactly one short answer string.
No explanation.
No units unless the unit is necessary to distinguish the answer.
Use plain fractions such as 17/10.
Use sqrt notation for radicals.
```

Prompting rules:

- Prefer short-answer prompts.
- Avoid chain-of-thought output in the final prediction.
- If using few-shot examples from `train.csv`, keep examples diverse:
  - Integer answer
  - Thai unit answer
  - Fraction or decimal answer
  - Diagram or geometry answer
  - LaTeX-style answer

## Postprocessing Requirements

The raw model output must be cleaned before writing `submission.csv`.

Postprocessing steps:

1. Convert output to string.
2. Strip outer whitespace.
3. Remove common answer prefixes:
   - `answer:`
   - `final answer:`
   - `คำตอบ:`
   - `คำตอบคือ`
   - `ตอบ`
4. If the model returns multiple lines, prefer the first line that looks like an answer.
5. Remove surrounding quotes.
6. Remove trailing punctuation such as `.`, `,`, `;`, `:`.
7. Convert Thai digits to Arabic digits.
8. Remove Markdown or code fence artifacts.
9. Normalize common LaTeX delimiters by removing `$`.
10. Keep Thai units only if the model output is otherwise ambiguous. Kaggle normalization removes many units.
11. If the cleaned answer is empty, use a fallback answer from the training prior.

Fallback rule:

- Use the most common simple answer from train, such as `1` or `2`, only when the model output is empty or unusable.
- Log every fallback case with image id and raw model output.

Postprocessing must preserve:

- Thai phrases when they are the full answer.
- Fractions such as `17/10`.
- Radicals such as `6sqrt3` or `sqrt(3)`.
- Negative numbers.
- Decimals.
- Choice labels if the expected answer is a choice label.

## Local Validation Plan

Use `train.csv` for local checks before predicting the test set.

Recommended validation:

- Create a small holdout split from train, for example 20-30%.
- Run the full inference pipeline on the holdout images.
- Compare both raw exact accuracy and Kaggle-normalized accuracy.
- Inspect errors manually by image type.

Track error categories:

- OCR failure on Thai text
- OCR failure on English text
- Small superscript missed
- Geometry or graph reasoning failure
- Shape or visual-choice reasoning failure
- Unit formatting mismatch
- Fraction or LaTeX formatting mismatch
- Model returned explanation instead of answer
- Model returned empty or unrelated text

## Submission Validation

Before saving the final CSV, the notebook must validate:

- `submission.csv` has exactly 420 rows.
- Columns are exactly `id,answer`.
- Every id from `sample_submission.csv` appears exactly once.
- No extra ids appear.
- Row order matches `sample_submission.csv`.
- No answer is null.
- No answer is an empty string after stripping whitespace.
- All answers are serializable as CSV strings.

If any validation check fails, the notebook must raise an error before submission.

## Experiment Log

Every serious run should be logged.

| Run | Model            | Setup     | Preprocessing | Prompt                   | Postprocessing | Local score | Public LB | Notes    |
| --- | ---------------- | --------- | ------------- | ------------------------ | -------------- | ----------- | --------- | -------- |
| 001 | LLaVA via Ollama | Colab T4  | raw RGB       | base short-answer prompt | basic cleanup  | n/a         | n/a       | baseline |

Log notes should include:

- Whether Ollama installed successfully.
- Whether the model was downloaded or loaded from dataset assets.
- Runtime per image.
- Number of empty or fallback predictions.
- Common postprocessing failures.
- Images requiring special preprocessing.

## Implementation Tasks

The Colab notebook should be organized into these sections:

1. Colab environment and path setup
2. Kaggle credential setup
3. Competition data download with Kaggle CLI/API
4. Data extraction to `/content/input/super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen/`
5. Colab system dependency setup, including `zstd` for Ollama extraction
6. Data loading
7. Data integrity checks
8. Optional train EDA summary
9. Image loading and preprocessing functions
10. Ollama/LLaVA setup
11. Prompt templates
12. Single-image inference smoke test
13. Train holdout validation
14. Test inference loop
15. Prediction postprocessing
16. Submission validation
17. Write `/content/submission.csv` and optionally copy it to `/content/working/submission.csv`
18. Save raw prediction logs under `/content/working/` for debugging

Required generated files:

```text
/content/submission.csv
/content/working/raw_predictions.csv
```

`raw_predictions.csv` should contain:

| Column            | Meaning                          |
| ----------------- | -------------------------------- |
| `id`              | Test image id                    |
| `image_path`      | Image path from `test.csv`       |
| `raw_prediction`  | Original model response          |
| `clean_answer`    | Postprocessed answer             |
| `prompt_name`     | Prompt template used             |
| `preprocess_name` | Preprocessing variant used       |
| `final_size`      | Prepared image size passed to VQA model, formatted as `widthxheight` |
| `runtime_seconds` | Wall-clock seconds spent on preprocessing and inference for the row |
| `inference_error` | Empty string for successful model calls, otherwise the timeout/request error used to trigger fallback |
| `used_fallback`   | Whether fallback answer was used |

## Acceptance Criteria

The implementation is complete when:

- The notebook runs end-to-end in Google Colab after Kaggle credentials are provided and competition rules are accepted.
- The notebook downloads the Kaggle competition archive with the Kaggle CLI or API, extracts it into `/content/input/super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen/`, and only then reads CSVs or images.
- The notebook produces `/content/submission.csv`.
- The submission file passes all validation checks.
- Raw predictions are saved for error analysis.
- At least one local smoke test on train images is shown.
- The chosen preprocessing, prompt, and postprocessing configuration is recorded in the experiment log.
