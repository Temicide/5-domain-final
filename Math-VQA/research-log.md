# Research Log

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-06-06 | bootstrap | Initialized autoresearch for replacing the weak Ollama/LLaVA Thai Math VQA pipeline on a single Colab Pro H100 80GB. Inspected notebook, helper modules, local data, and old raw predictions. |
| 2 | 2026-06-06 | bootstrap | Literature/model scan selected Qwen3-VL-32B-Instruct as the first H100 inference target, Qwen2.5-VL-32B/72B as proven fallback family, PaddleOCR-VL as auxiliary Thai/formula OCR, and QLoRA/LoRA as the controlled fine-tuning route. |
| 3 | 2026-06-06 | inner-loop | Baseline failure replay: old LLaVA test log had 420 rows, 81 long answers over 40 chars, and only 15 recorded fallbacks. Stricter cleaner replay marks 75 fallbacks and changes 85 answers while extracting declared answers from some verbose responses. |
| 4 | 2026-06-06 | inner-loop | Implemented stricter postprocessing and regenerated `notebooks/thai_math_vqa_ollama_llava.ipynb`; all 37 Math-VQA tests pass. This is a low-risk baseline hardening independent of the future Qwen replacement. |
| 5 | 2026-06-06 | protocol | Locked H1 protocol: evaluate Qwen3-VL-32B-Instruct plus optional PaddleOCR-VL prompt augmentation on a train holdout before running test inference. |
| 6 | 2026-06-06 | inner-loop | Added generated H100/Qwen-VL Colab notebook at `notebooks/thai_math_vqa_h100_qwen_vl.ipynb`. It defaults to `Qwen/Qwen3-VL-32B-Instruct`, includes holdout validation, 4-bit and flash-attention switches, and preserves the validated submission/raw log contract. All 38 Math-VQA tests pass. |
| 7 | 2026-06-06 | inner-loop | Implemented H2 setup in the generated H100 notebook: `USE_PADDLEOCR_VL=1` loads `PaddlePaddle/PaddleOCR-VL`, extracts OCR/formula context with caching, and appends it to the Qwen prompt while preserving image-only baseline defaults. All 38 Math-VQA tests pass. |
