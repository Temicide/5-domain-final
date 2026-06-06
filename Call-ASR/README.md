# Thai Call Center ASR

This project generates a Google Colab-ready Thai call-center ASR submission CSV for the Kaggle competition `individual-test-thai-call-center-asr`.

## Local Setup

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pip install -e .
python3 -m pytest -v
```

## Audit Competition Audio

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m call_asr.audit_audio \
  --audio-dir /Users/temicide/Documents/5_domain_final/Call-ASR/data/individual-test-thai-call-center-asr/audio_final/audio \
  --output-csv /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/audio_inventory.csv \
  --failures-jsonl /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/decode_failures.jsonl
```

Expected local audit summary:

```text
Audited 6261 WAV files
Decode failures: 0
Prefix counts: {'AU': 400, 'BCH': 240, 'FD': 11, 'INT': 1080, 'RSP': 720, 'SDB': 3330, 'TT': 480}
```

## Run Inference

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m call_asr.infer \
  --model-name typhoon-ai/typhoon-whisper-large-v3 \
  --normalization-policy single_space \
  --chunk-length-s 30 \
  --batch-size 4
```

Expected local artifacts:

```text
/Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/predictions_typhoon-ai__typhoon-whisper-large-v3_single_space.csv
/Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/run_typhoon-ai__typhoon-whisper-large-v3_single_space.jsonl
```

## Colab Notebook

Upload `/Users/temicide/Documents/5_domain_final/Call-ASR/notebooks/colab_submission.ipynb` into Google Colab with GPU enabled. The editable script source is kept at `/Users/temicide/Documents/5_domain_final/Call-ASR/notebooks/colab_submission.py`. Provide Kaggle credentials with one of these methods before running the notebook:

- Upload `kaggle.json` to `/content/kaggle.json`.
- Store `KAGGLE_USERNAME` and `KAGGLE_KEY` in Colab secrets or environment variables.

The notebook writes credentials to `~/.kaggle/kaggle.json` with restrictive permissions, never prints credential values, downloads the competition data with Kaggle CLI/API, extracts it to `/content/input/individual-test-thai-call-center-asr`, validates the sample submission schema and audio coverage, runs resumable inference, and writes:

```text
/content/submission.csv
```

Intermediate Colab artifacts are written under `/content/working`.

Do not run kaggle competitions submit. Manual upload through the Kaggle UI is the allowed boundary for the generated CSV.
