# PaddleOCR-VL

- Source: https://huggingface.co/PaddlePaddle/PaddleOCR-VL
- Technical report: https://arxiv.org/abs/2510.14528
- Relevance: auxiliary OCR/formula/chart parser for Thai math images.

Summary:

PaddleOCR-VL is a 0.9B document-parsing VLM with a NaViT-style dynamic-resolution visual encoder. Its model card says it supports 109 languages including Thai and recognizes complex elements such as text, tables, formulas, and charts.

Implication for this project:

Use PaddleOCR-VL as an auxiliary channel: extract Thai/English text and formulas, pass that text into the Qwen prompt, and still include the original image. This targets the current LLaVA failure mode where the model often asks for a clearer image instead of solving.

