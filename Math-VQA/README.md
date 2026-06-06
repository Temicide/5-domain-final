# Thai Math VQA Colab Pipeline

This project builds a Google Colab-ready notebook for the Super AI Engineer SS6 Thai Math VQA challenge. The notebook downloads the competition data with the Kaggle CLI, uses LLaVA through Ollama, produces `/content/math-vqa-output/submission.csv`, and saves `/content/math-vqa-output/raw_predictions.csv` for error analysis.

## Local Setup

```bash
cd Math-VQA
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python scripts/build_notebook.py
```

If your shell already has `python` pointing at a usable Python 3.10+ interpreter, `python scripts/build_notebook.py` works too.

The generated notebook is `notebooks/thai_math_vqa_ollama_llava.ipynb`.

## Colab Setup

1. Upload or open the generated notebook in Google Colab.
2. Use a GPU runtime.
3. Accept the competition rules on Kaggle.
4. Provide Kaggle credentials through one of these options:
   - Set `KAGGLE_USERNAME` and `KAGGLE_KEY` as environment variables.
   - Place `kaggle.json` at `/root/.kaggle/kaggle.json`.
   - Upload `kaggle.json` when the notebook prompts through `google.colab.files.upload()`.
5. Enable internet for the first Kaggle data download, `zstd` apt install, Ollama install, and `llava` pull.
6. Run the notebook.
7. Submit `/content/math-vqa-output/submission.csv` manually to Kaggle.

The notebook defaults to `OLLAMA_REQUEST_TIMEOUT=600` and `OLLAMA_RETRIES=1`. If Colab still times out on slow images, set a larger notebook environment value such as `OLLAMA_REQUEST_TIMEOUT=900`. Per-image timeout/request failures are logged in `raw_predictions.csv`, assigned the training-prior fallback answer, and do not stop the submission loop.

The notebook uses:

```text
pip install kaggle
kaggle competitions download -c super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen -p /content/math-vqa-data -o
apt-get update
apt-get install -y --no-install-recommends zstd
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull llava
```

If the runtime cannot install `zstd`, or if internet/model download is unreliable, package the Ollama model cache as a Kaggle dataset and adjust the setup cell to point Ollama at the attached model directory.

## Hugging Face Fallback

If Ollama cannot run in the Colab environment, download or attach a compatible Hugging Face vision-language model such as `llava-hf/llava-1.5-7b-hf` or `Qwen/Qwen2.5-VL-3B-Instruct`. Replace the notebook call to `query_llava(...)` with a Transformers image-to-text call that returns one raw answer string. Keep these pieces unchanged:

- `clean_model_answer(...)`
- `PredictionRecord`
- `write_outputs(...)`
- `submission.csv` validation
- `raw_predictions.csv` logging

## Output Contract

The final submission must contain exactly these columns:

```csv
id,answer
```

The row order must match `sample_submission.csv`, ids are strings, answers are strings, and empty answers are rejected before the file is written. Raw prediction logs include the prompt name, preprocessing variant, final prepared image size, raw model response, cleaned answer, and fallback flag.
