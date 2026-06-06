# H100 VLM Research Brief: Thai Math VQA

Date: 2026-06-06

## Recommendation

Replace the Ollama/LLaVA path with a Qwen VLM pipeline:

1. First run `Qwen/Qwen3-VL-32B-Instruct` on a fixed train holdout with high-resolution image handling.
2. Add PaddleOCR-VL extracted Thai/formula text to the prompt and compare against image-only Qwen.
3. Use Qwen2.5-VL-32B as the fallback if Qwen3-VL is unstable in Colab.
4. Try LoRA/QLoRA only after the validation harness is locked; start with Qwen3-VL-8B, not 32B/72B.

## Why LLaVA Should Be Replaced

The current raw test log has many accepted non-answers. The old cleaner recorded 15 fallbacks, but replaying the same outputs with stricter postprocessing marks 75 fallbacks and changes 85 answers. This is not just a timeout problem; it is a model and answer-extraction problem.

## Why Qwen3-VL / Qwen2.5-VL

Qwen3-VL is current and supports direct Transformers/vLLM image-text inference. Qwen2.5-VL has published OCR/math/document benchmark evidence and dynamic-resolution image processing. Qwen2.5-VL-32B reports MathVista 74.7, MathVision 40.0, DocVQA 94.8, and OCRBenchV2 57.2/59.1 on its model card.

## Why OCR-Augmented Prompting

Thai Math VQA images are mostly wide and text-heavy. In this dataset, 194/280 train and 291/420 test images have width/height greater than 4. PaddleOCR-VL supports Thai and complex document elements like formulas/charts, so it can provide a text channel while the VLM still reasons over the original image.

## H100 Use

Use the H100 for a bigger, higher-resolution model first:

- `Qwen/Qwen3-VL-32B-Instruct`
- BF16 + flash attention if it fits
- bounded visual-token budget, starting around `2048*28*28`
- deterministic generation, max 32 new tokens

Fine-tuning is second priority. With only 280 labels, LoRA can overfit. If needed, train Qwen3-VL-8B adapters under 5-fold cross-validation, then train on all labels only for the final run.

## Immediate Code Change Already Made

Two code changes are already made:

- `notebooks/thai_math_vqa_h100_qwen_vl.ipynb` was added as a generated H100/Qwen notebook.
- The H100/Qwen notebook now has an optional OCR augmentation switch: set `USE_PADDLEOCR_VL=1` to add PaddleOCR-VL OCR/formula context.
- The existing Ollama/LLaVA notebook was regenerated with stricter postprocessing.
- rejects common unreadable/refusal strings like "too small", "blurry", "please provide"
- extracts declared final answers from verbose responses
- keeps the output contract unchanged
- all 38 Math-VQA tests pass

## Next Experiment

Run `notebooks/thai_math_vqa_h100_qwen_vl.ipynb` on Colab H100 twice: first with `USE_PADDLEOCR_VL=0` for the image-only Qwen baseline, then with `USE_PADDLEOCR_VL=1` for OCR/formula augmentation. Record both holdout scores in `experiments/h1-qwen3vl32b-ocr-augmented/results/`. The locked H1 protocol is in `experiments/h1-qwen3vl32b-ocr-augmented/protocol.md`.
