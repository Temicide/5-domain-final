# Individual Test: Thai Call Center ASR - Competition Spec

Competition: https://www.kaggle.com/competitions/individual-test-thai-call-center-asr

Last updated: 2026-06-06

## Objective

Build the lowest-error Thai ASR submission for a Kaggle test-only call-center
transcription competition. The deliverable is a CSV with one transcript per WAV
file. The visible data does not include labels, so model selection must be driven
by external validation sets, careful audio/data inspection, public leaderboard
feedback, and conservative post-processing.

## Colab Notebook Requirement

The deliverable must be a Google Colab-ready solution. The notebook must:

- Run end-to-end in Google Colab.
- Install or import the Kaggle CLI/Python API inside Colab as needed.
- Accept Kaggle credentials either from an uploaded `kaggle.json` file or from
  Colab secrets/environment variables, write them only to the expected local
  credentials path with restrictive permissions, and never print credential
  values.
- Download the competition data in Colab with the Kaggle CLI or Kaggle API
  before any file reads, then extract the downloaded archive under
  `/content/input/individual-test-thai-call-center-asr/`.
- Read competition files from `/content/input/...` in Colab, with local path
  fallbacks under this project only for development.
- Generate `/content/submission.csv` using the sample submission row order and
  schema. Intermediate artifacts should default to `/content/working/`.
- Validate the generated CSV before saving or before declaring the run complete.
- Not submit through the Kaggle API, Kaggle CLI, or any other automated submission mechanism.

Manual upload to Kaggle is allowed outside the notebook, but implementation must
stop after generating `submission.csv`.

## Verified Local Data

Local competition directory:

```text
Call-ASR/data/individual-test-thai-call-center-asr/
  audio_final/audio/*.wav
  sample_submission.csv
```

Current local counts:

- `6261` WAV files.
- `6261` submission rows.
- `sample_submission.csv` has exactly two columns: `file_name,text`.
- Every submission `file_name` has a matching local WAV file.

Filename-family counts:

| Prefix | Count | Likely role |
|---|---:|---|
| `SDB` | 3330 | bulk short utterances |
| `INT` | 1080 | intent/interview-like short utterances |
| `RSP` | 720 | response utterances |
| `TT` | 480 | task/test template utterances |
| `AU` | 400 | longer call audio |
| `BCH` | 240 | short call snippets |
| `FD` | 11 | fixed/demo responses |

Sample media probe across 271 stratified files:

- Codec: PCM signed 16-bit little-endian.
- Channels: mono.
- Sample rates observed: 16 kHz and 24 kHz.
- `AU` files are much longer: sampled median about 34 seconds, max about 129 seconds.
- Most non-`AU` files are short: sampled medians roughly 3 to 12 seconds depending on prefix.

Implementation implication: do not assume a single sample rate or fixed duration.
Decode through ffmpeg/torchaudio/librosa, resample to the model-required rate,
and log per-file decode failures.

## Kaggle Context

The Kaggle skill was used to check credential state and competition interaction
options. The local environment has legacy Kaggle credentials configured, but the
`kaggle` CLI/Python package is not installed on the active `python3` path. The
existing local data and previous API-derived metadata indicate:

- Task: Thai automatic speech recognition for call-center audio.
- Evaluation: edit-distance based error rate, described by Kaggle as
  `(S + D + I) / N`; lower is better.
- Competition pages previously exposed evaluation/rules/files/leaderboard
  metadata, but not all overview/data-description fields.
- Existing visible baseline metadata in the old spec showed a baseline score of
  `4.53538`; treat this only as a coarse sanity target until verified again.

Kaggle package setup command for local development if more API work is needed:

```bash
python3 -m pip install --user kaggle
python3 -m kaggle competitions files -c individual-test-thai-call-center-asr
```

Do not print or commit Kaggle credentials.

## Submission Contract

Use the sample submission schema exactly:

```csv
file_name,text
RSP_101_audio.wav,สวัสดีค่ะ
```

Rules for submission generation:

1. Preserve row order from `sample_submission.csv`.
2. Preserve `file_name` exactly, including `.wav`.
3. Put the final transcript in `text`.
4. Never leave `text` empty except for files confidently detected as silence.
5. Save the final Colab output to `/content/submission.csv`.
6. Optionally save Colab candidate artifacts under `/content/working/` and local
   development candidates under a timestamped path such as
   `Call-ASR/data/submissions/sub_YYYYMMDD_HHMM_model-note.csv`.
7. Do not call `kaggle competitions submit` or any Kaggle API submission endpoint from the notebook.

## Primary Scoring Hypothesis

The best score will come from an ensemble of Thai-specialized Whisper-family
models plus data-specific normalization, not from a single generic ASR pass.

Reasoning:

- The competition has no labels, so direct supervised adaptation is unavailable.
- Thai call-center speech is likely conversational, noisy, and partly
  code-switched; generic multilingual ASR can miss Thai-specific orthography and
  spacing.
- Whisper-derived Thai models are currently the strongest accessible starting
  point for Thai ASR. Public model cards report strong Thai-specific training
  and benchmarks:
  - Typhoon Whisper Large v3 is Thai-specific, built on Whisper Large v3, and
    trained on about 11k hours / 10M Thai audio samples.
    Source: https://huggingface.co/typhoon-ai/typhoon-whisper-large-v3
  - Thonburian Whisper models are Thai Whisper fine-tunes using Common Voice,
    Gowajee, Thai Elderly Speech, and Thai dialect data; the large-v3 model is
    reported as the highest-accuracy option in that family.
    Source: https://github.com/biodatlab/thonburian-whisper
  - Pathumma/NECTEC-derived noisy Thai Whisper variants report strong CER on
    Gowajee and noisy/spontaneous Thai benchmarks.
    Source: https://huggingface.co/PogusTheWhisper/Pathumma-whisper-th-large-v3-natural-noise-finetuned
  - Older Thai Wav2Vec2 work shows that Thai-specific ASR plus a language model
    can improve robustness, but the model family is likely a secondary ensemble
    member rather than the main engine.
    Source: https://arxiv.org/abs/2208.04799

## Winning Strategy

### 1. Build a Reproducible Inference Harness

Create one script that can run any model over all `sample_submission.csv` rows:

```text
src/
  infer.py              # model runner, batching, resume, logging
  normalize_text.py     # Thai-safe normalization variants
  score_proxy.py        # CER/WER/edit-distance evaluation
  ensemble.py           # combine candidate transcripts
  audit_audio.py        # duration/sample-rate/decode report
```

Minimum harness features:

- Resume from partial outputs.
- Per-file JSONL logs with model name, decode parameters, runtime, raw text,
  normalized text, confidence/probability if available, and errors.
- Batch inference on GPU when available; CPU fallback for short/debug runs.
- ffmpeg-based resampling to 16 kHz for Whisper-family models.
- Separate raw and normalized transcript columns in intermediate artifacts.

### 2. Establish Proxy Validation Before Tuning

Because hidden labels are unavailable, create external validation sets that mimic
the competition:

- Common Voice Thai.
- Gowajee via SEACrowd if available.
- FLEURS Thai.
- Thai Elderly Speech.
- LOTUSDIS or other conversational/noisy Thai corpora if licenses permit.
- Any public Thai call-center, customer-service, finance, banking, or telephone
  speech data allowed by competition rules.

Build two validation transforms:

- `clean`: original public audio.
- `call_center_degraded`: mono, 8/16 kHz telephone bandpass, light background
  noise, gain variation, compression artifacts, and optional VAD chunking.

Tune decisions on proxy CER first, then use Kaggle public leaderboard only to
choose among a small number of pre-registered variants.

### 3. Run Strong Baselines

Priority order:

1. `typhoon-ai/typhoon-whisper-large-v3`
2. `biodatlab/whisper-th-large-v3-combined` or the best available Thonburian
   large-v3 checkpoint
3. `PogusTheWhisper/Pathumma-whisper-th-large-v3-natural-noise-finetuned`
4. `openai/whisper-large-v3` as a generic fallback/reference
5. Thai Wav2Vec2/XLSR with an external Thai LM as a diversity model

For each Whisper model, sweep only a small controlled grid:

- chunk length: 20, 30, 45 seconds for long `AU` files.
- beam size: 1, 3, 5.
- temperature: 0.0 first; fallback schedule only for low-confidence output.
- language/task prompt: force Thai transcription.
- condition-on-previous-text: compare on/off for long `AU` files.
- VAD: compare no VAD versus conservative VAD chunking.

Avoid broad random sweeps; without labels, they overfit leaderboard noise.

### 4. Normalize for the Metric, Not for Human Readability

Create multiple Thai-safe normalization variants and test them on proxy data and
leaderboard submissions:

- `raw`: model output minimally stripped.
- `thai_chars_only_light`: remove punctuation and emoji, keep Thai, digits,
  English letters, and spaces.
- `no_spaces`: remove all whitespace. This is often strong if the scorer is
  character-level and references are not space-sensitive.
- `single_space`: collapse whitespace to a single space.
- `spoken_numbers`: compare Arabic digits versus Thai words only if proxy data
  shows a consistent reference style.
- `remove_fillers`: remove only known filler/noise tokens if validated.

Critical caution: Thai spacing is inconsistent across ASR models and corpora.
The first few Kaggle submissions should explicitly compare `raw`, `single_space`,
and `no_spaces`. Do not assume human-readable spacing is optimal for edit
distance.

### 5. Segment Long Files Conservatively

For short non-`AU` files, transcribe whole files.

For `AU` files:

- Start with Whisper chunking at 30 seconds.
- Compare against VAD segmentation with 0.2 to 0.5 second padding.
- Avoid aggressive VAD that clips Thai syllables at boundaries.
- Stitch segments with a single configurable separator, then pass through the
  normalization variant being tested.

If `condition_on_previous_text=True` causes repetition loops, disable it for
long files and rely on overlap/deduplication.

### 6. Ensemble Transcripts

Use a small ensemble only after the individual baselines exist.

Candidate approaches:

- Per-file confidence selection: choose the model with the best average token
  log probability / lowest compression ratio / lowest no-speech probability.
- Prefix-specific selection: if one model wins on short `RSP`/`SDB` proxy data
  and another wins on noisy/long `AU`, route by filename prefix.
- ROVER-style voting at character or syllable/token level for outputs that are
  close in length.
- LLM/text-repair only as a final conservative pass for obvious ASR artifacts;
  it must not invent content. Use it on proxy validation before any Kaggle use.

Do not ensemble by majority at the word level unless Thai tokenization is fixed
and validated.

### 7. Use Public Leaderboard as a Limited Oracle

Proposed submission sequence:

1. Baseline `typhoon-large-v3 + raw/light strip`.
2. Same transcripts with `no_spaces`.
3. Best Thonburian/Pathumma model with the better normalization from step 1-2.
4. Prefix-routed ensemble.
5. Final selected ensemble with the most stable normalization.

Record every submission:

```text
submission_id
timestamp
model(s)
decode params
normalization
score
diff from previous
decision
```

Do not spend leaderboard attempts on tiny changes unless they are supported by
proxy validation or error analysis. The notebook itself must only generate CSV
artifacts; any upload/submission step is manual and outside the notebook.

## Autoresearch Experiment Plan

Optimization target: minimize hidden Kaggle edit-distance score.

Proxy metric: Thai CER on external validation sets using the exact same
normalization candidates used for submission.

Initial hypotheses:

| ID | Hypothesis | Prediction | Priority |
|---|---|---|---|
| H1 | Thai-specialized Whisper Large v3 beats generic Whisper Large v3. | Lower proxy CER and leaderboard score. | P0 |
| H2 | Removing Thai spaces improves character-level scoring. | `no_spaces` beats `raw` or `single_space` if scorer is char-heavy. | P0 |
| H3 | Noisy/conversational fine-tunes beat clean-speech Thai models on `AU` and degraded proxy audio. | Prefix-routed ensemble improves over single model. | P1 |
| H4 | Conservative VAD helps long `AU` files but hurts short clips. | VAD only improves long-file proxy subsets. | P1 |
| H5 | Character/syllable-level voting improves over confidence selection. | Ensemble reduces substitutions without adding insertions. | P2 |
| H6 | Thai LM/spell correction helps domain terms but risks hallucination. | Only improves when constrained to edit likely ASR artifacts. | P2 |

Inner-loop protocol:

1. Lock one hypothesis and a small decode/normalization grid.
2. Run proxy validation.
3. Generate a candidate full submission only if proxy results are plausible.
4. Upload manually, if chosen, and record score outside the notebook.
5. Update the experiment table and choose the next hypothesis.

Outer-loop reflection every 3 to 5 submissions:

- Which model family wins by prefix and audio condition?
- Which normalization style is consistently best?
- Are leaderboard gains aligned with proxy gains?
- Is the strategy still improving, or are leaderboard attempts overfitting public LB?

## Implementation Milestones

1. `audit_audio.py`: full data inventory with duration/rate/prefix/decode status.
2. `normalize_text.py`: reversible named normalization policies.
3. `infer.py`: one-model full inference with resume support.
4. Proxy validation download/build script.
5. Baseline runs for Typhoon, Thonburian, Pathumma, and OpenAI Whisper.
6. Normalization sweep on proxy validation and 2 to 3 Kaggle submissions.
7. Prefix-specific and confidence-based ensemble.
8. Final reproducibility report with model versions, data sources, and scores.

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Hidden scorer is word-level rather than character-level. | `no_spaces` may fail badly. | Generate `raw/single_space/no_spaces` CSVs early and manually upload only if needed to infer behavior. |
| Public leaderboard is small or noisy. | Overfitting. | Use proxy validation and spend manual leaderboard attempts only on coarse variants. |
| Rules restrict external APIs or non-public models. | Disqualification risk. | Use public open models/datasets unless rules explicitly allow more. |
| Thai normalization removes meaningful text. | Deletions increase. | Keep raw outputs and compare named policies. |
| Long `AU` files repeat or truncate. | Large edit distance on hardest files. | Tune chunking, VAD padding, and repetition filters separately for `AU`. |
| Code-switching with English/digits appears. | Thai-only cleanup can delete correct content. | Preserve English letters and digits unless validation says otherwise. |

## Immediate Next Actions

1. Install/activate Kaggle CLI and re-query the competition pages/files/rules.
2. Run full audio audit and write `data/audio_inventory.csv`.
3. Implement `normalize_text.py` and `score_proxy.py`.
4. Run the first 50-file smoke test with Typhoon Whisper Large v3.
5. Build proxy validation from public Thai speech corpora.
6. Run H1 and H2 before any broad experimentation.
