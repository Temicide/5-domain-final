# Research Findings

## Research Question

Which open-weight vision-language pipeline should replace the current Ollama/LLaVA notebook for best Thai math VQA accuracy on a single Colab Pro H100 80GB runtime?

## Current Understanding

The current pipeline fails primarily because it uses an outdated general LLaVA path through Ollama, with no local validation loop and weak filtering of non-answer generations. The old 420-row raw prediction log contains many readable symptoms: "too small", "blurry", "not possible", "please provide", and long step-by-step explanations being accepted as final answers.

The most defensible next pipeline is not "fine-tune first." With only 280 labeled examples, the first bet should be a stronger high-resolution VLM and a locked validation harness. Qwen3-VL-32B-Instruct is the best first H100 target because it is current, open-weight, and directly supports image-text chat through Transformers/vLLM. Qwen2.5-VL-32B/72B remains a strong fallback because its model cards report strong MathVista, MathVision, DocVQA, OCRBench, and CC-OCR results, and its dynamic-resolution processing matches the dataset's wide small-text images.

PaddleOCR-VL is worth adding as an auxiliary channel, not as the only solver. It supports Thai among 109 languages and is designed for text, formulas, charts, and document parsing. The likely winning prompt should pass both the image and OCR text/formula extraction, then force a short answer.

## Key Results

- Existing test output replay: old `raw_predictions-1.csv` had 420 rows, 81 answers longer than 40 characters, and 15 recorded fallbacks.
- New cleaner replay: fallback detection rises from 15 to 75 rows and 85 cleaned answers change. Examples recovered include verbose outputs with final answers `-2` and `2.1886`.
- Data shape: 194/280 train images and 291/420 test images have width/height greater than 4, so fixed square resizing is a bad default.
- Runnable H100 notebook: `notebooks/thai_math_vqa_h100_qwen_vl.ipynb` now builds from `scripts/build_h100_qwen_notebook.py` and defaults to `Qwen/Qwen3-VL-32B-Instruct`.
- H2 is now runnable: set `USE_PADDLEOCR_VL=1` to load `PaddlePaddle/PaddleOCR-VL`, extract OCR/formula context, cache it under the output directory, and append it to the Qwen prompt.
- Tests: all 38 Math-VQA tests pass after postprocess hardening and H100 notebook generation.

## Patterns and Insights

The dataset is OCR-heavy, but not OCR-only. Many questions include diagrams, shapes, graph axes, answer choices, formulas, and Thai units. A pure OCR-to-text solver will miss visual context; a pure VLM without OCR help can miss small Thai text. The likely best architecture is "high-resolution VLM sees image + auxiliary OCR text + answer-normalized validation."

The H100 should be used to increase model quality and visual-token budget, not just run LLaVA faster. For a single 80GB H100, Qwen3-VL-32B BF16/FP8 inference or 4-bit fallback is the right first use. Adapter training should start with Qwen3-VL-8B or Qwen2.5-VL-7B/8B under cross-validation; 32B/72B adapter training is a later experiment if validation proves smaller adapters underperform.

## Lessons and Constraints

- Do not trust public test inference without a train holdout score; the repo already has local train labels and normalized accuracy helpers.
- Do not train aggressively on all 280 labels before validating. Use fixed holdout or 5-fold CV, then train on all labels only for final submission.
- Do not crop by default. Preserve aspect ratio and use model visual-token controls (`min_pixels`, `max_pixels`) or explicit wide-image tiling.
- Do not accept long natural-language responses as answers. The notebook now rejects common refusal/unclear-image text and extracts declared final answers.
- Official Qwen fine-tuning guidance says training resolution is critical and notes Qwen2.5-VL-32B training in their script expects 8x80GB GPUs; single-H100 32B experiments should therefore use inference first or QLoRA/adapters with careful memory checks.

## Open Questions

- What is the fixed-seed train holdout accuracy for current LLaVA after cleaner hardening?
- Does Qwen3-VL-32B fit reliably in Colab H100 BF16 at a useful visual-token budget, or should the notebook default to FP8/vLLM or 4-bit?
- Does PaddleOCR-VL improve Thai/formula extraction enough to justify its extra runtime?
- Does Qwen3-VL-8B LoRA improve held-out normalized accuracy, or does it overfit output style?
- Which image variants help most: raw high-token, upscaled low-res, contrast, or wide tiling?

## Optimization Trajectory

| Run | Change | Metric |
|---|---|---|
| run_001 | Replayed old LLaVA outputs through stricter cleaner | fallback_count 15 -> 75; changed_answers=85; normalized_accuracy not measurable on test labels |
| run_002 | Added generated H100/Qwen-VL notebook | notebook contract test passes; validation metric pending Colab H100 run |
| run_003 | Added optional PaddleOCR-VL OCR/formula prompt augmentation | notebook contract test passes; validation metric pending Colab H100 run |
