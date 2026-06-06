# H1 Protocol: Qwen3-VL-32B + OCR-Augmented Prompting

Status: confirmatory protocol locked on 2026-06-06.

## Hypothesis

Qwen3-VL-32B-Instruct with high visual-token budget, flash attention, and answer-only prompting will outperform the current Ollama/LLaVA pipeline on Thai Math VQA local train-holdout normalized accuracy. Adding PaddleOCR-VL text/formula extraction should reduce unreadable/refusal failures.

## Motivation

The old LLaVA/Ollama raw log accepts many long non-answer strings. Replay with the new cleaner marks 75 likely fallbacks out of 420 test rows. The data is also extremely wide: 194/280 train and 291/420 test images have width/height greater than 4, so a model with dynamic/high-resolution visual handling is needed.

## Evaluation

Primary metric:

- `normalized_accuracy(predictions, truths)` from `math_vqa.evaluation`

Secondary metrics:

- fallback count
- long answer count (`len(clean_answer) > 40`)
- median runtime per image
- peak GPU memory
- per-image error notes for OCR/diagram/wide cases

Validation design:

- Use a fixed seed train split first: 224 train-like examples, 56 holdout examples.
- If Qwen3-VL-32B looks promising, run 5-fold CV before committing to a final test submission.
- Only after validation is locked should the final model be run on all 420 test images.

## Conditions

Run A: current LLaVA/Ollama notebook after cleaner hardening on the same holdout.

Run B: Qwen3-VL-32B-Instruct, image only, BF16/flash-attention if memory allows.

Run C: Qwen3-VL-32B-Instruct, image plus PaddleOCR-VL OCR/formula text.

Run D: fallback Qwen2.5-VL-32B-Instruct if Qwen3-VL is unavailable or unstable in Colab.

## Prediction

Run B should reduce refusal/long-answer rate sharply. Run C should improve Thai/formula-heavy wide images. If Run C does not improve accuracy, PaddleOCR noise or prompt distraction is likely, and OCR should be used selectively only for images where the base VLM indicates uncertainty.

## H100 Configuration

- Prefer `torch.bfloat16`, `attn_implementation="flash_attention_2"`, and `device_map="auto"`.
- Start with `max_pixels=2048*28*28`; reduce to `1280*28*28` if OOM, increase for wide/diagram cases if memory allows.
- Use deterministic generation for scoring: temperature 0.0 or equivalent greedy decoding, max new tokens 32.
- If BF16 OOMs, retry Qwen3-VL-32B with 4-bit bitsandbytes or use Qwen2.5-VL-32B. Do not silently fall back to LLaVA.

## Result Recording

Save:

- `experiments/h1-qwen3vl32b-ocr-augmented/results/holdout_predictions_{run}.csv`
- `experiments/h1-qwen3vl32b-ocr-augmented/results/metrics_{run}.json`
- updated `findings.md`
- updated `research-state.yaml`

