# Literature Survey: Thai Math VQA H100 Upgrade

## Model Direction

The strongest open-weight direction is Qwen-family VLMs rather than the current Ollama `llava` default.

- Qwen3-VL is the current Qwen vision-language family. The technical report describes strong multimodal reasoning, long context, and visual-math benchmarks such as MathVista and MathVision. The Hugging Face model card for `Qwen/Qwen3-VL-32B-Instruct` gives direct Transformers, vLLM, and SGLang usage paths.
- Qwen2.5-VL remains a robust fallback. Its 32B model card reports MathVista 74.7, MathVision 40.0, DocVQA 94.8, OCRBenchV2 57.2/59.1, and strong text-side MATH 82.2. The 72B card reports Qwen2.5-VL's strengths on texts, charts, icons, graphics, layouts, structured outputs, and dynamic-resolution processing.
- PaddleOCR-VL is a targeted OCR/document parser. Its model card says it supports 109 languages including Thai and handles text, tables, formulas, and charts. This directly addresses the old LLaVA failure mode where the model says the image is too blurry or unreadable.
- QLoRA provides the memory-efficient adapter training rationale if we need single-H100 fine-tuning after zero-shot/few-shot validation.

## Gaps

No source gives a direct Thai Math VQA competition score. The path has to be empirical:

1. Lock a train holdout evaluation.
2. Run Qwen3-VL-32B zero-shot/few-shot high-resolution inference.
3. Add PaddleOCR-VL text/formula context.
4. Try LoRA/QLoRA only if validation says prompt/OCR is not enough.

## Sources

- Qwen3-VL Technical Report: https://arxiv.org/abs/2511.21631
- Qwen3-VL-32B-Instruct model card: https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct
- Qwen3-VL official repository and fine-tuning framework: https://github.com/QwenLM/Qwen3-VL
- Qwen2.5-VL Technical Report: https://arxiv.org/abs/2502.13923
- Qwen2.5-VL-32B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct
- Qwen2.5-VL-72B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct
- NVIDIA NeMo Megatron Bridge Qwen2.5-VL fine-tuning docs: https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/qwen/qwen2.5-vl.html
- PaddleOCR-VL model card: https://huggingface.co/PaddlePaddle/PaddleOCR-VL
- QLoRA paper: https://arxiv.org/abs/2305.14314

