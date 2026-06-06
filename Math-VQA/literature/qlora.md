# QLoRA

- Source: https://arxiv.org/abs/2305.14314
- Relevance: memory-efficient fine-tuning route on one H100.

Summary:

QLoRA fine-tunes adapters through a frozen 4-bit quantized base model, reducing memory enough to fine-tune very large models on a single GPU while preserving much of full fine-tuning quality.

Implication for this project:

If zero-shot/few-shot Qwen3-VL is not enough, start adapter experiments with Qwen3-VL-8B or Qwen2.5-VL-7B/8B under cross-validation. Use 32B QLoRA only after smaller adapters show a clear validation gain, because the official Qwen fine-tuning README and NVIDIA docs indicate 32B+ VL training is materially heavier than one-GPU inference.

