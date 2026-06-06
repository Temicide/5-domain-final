# Qwen3-VL

- Source: https://arxiv.org/abs/2511.21631
- Model card: https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct
- Relevance: first candidate for H100 inference.

Summary:

Qwen3-VL is the current Qwen VLM family. The arXiv abstract emphasizes stronger pure-text understanding, native 256K-token long context, and advanced multimodal reasoning over images and videos, including visual-math benchmarks such as MathVista and MathVision. The 32B model card provides direct Transformers usage and recommends flash attention for acceleration and memory saving. It also supports vLLM and SGLang serving paths.

Implication for this project:

Use `Qwen/Qwen3-VL-32B-Instruct` as H1. It should replace Ollama/LLaVA for train-holdout inference. On H100, start with BF16 plus flash attention and a bounded visual-token budget; fall back to 4-bit or FP8/vLLM if BF16 OOMs.

