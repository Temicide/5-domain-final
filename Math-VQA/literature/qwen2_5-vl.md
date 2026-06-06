# Qwen2.5-VL

- Technical report: https://arxiv.org/abs/2502.13923
- 32B model card: https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct
- 72B model card: https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct
- Relevance: proven fallback and benchmark anchor.

Summary:

Qwen2.5-VL introduced dynamic-resolution processing and reports strong document, diagram, chart, OCR, and visual-math capabilities. The 32B model card reports MathVista 74.7, MathVision 40.0, DocVQA 94.8, OCRBenchV2 57.2/59.1, and text MATH 82.2. The 72B model card highlights text/chart/icon/layout understanding, structured outputs, and visual localization; it reports MathVista 74.8, MathVision 38.1, DocVQA 96.4, and OCRBenchV2 61.5/63.7.

Implication for this project:

If Qwen3-VL code or memory is unstable in Colab, `Qwen/Qwen2.5-VL-32B-Instruct` is the most practical fallback. `Qwen/Qwen2.5-VL-72B-Instruct` is useful for 4-bit inference or external serving but is too large for comfortable single-H100 BF16 training.

