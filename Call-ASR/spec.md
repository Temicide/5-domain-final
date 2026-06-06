# Individual Test: Thai Call Center ASR

Source: https://www.kaggle.com/competitions/individual-test-thai-call-center-asr

Retrieved with the Kaggle API on 2026-06-06. The API exposed the evaluation
page, rules page, file listing, and leaderboard metadata. It did not expose
overview, data-description, timeline, prize, or FAQ page content for this
competition.

## Competition Summary

This is an individual Kaggle competition for Thai automatic speech recognition
(ASR) on call-center audio. The goal is to generate accurate Thai transcripts
for the provided `.wav` test audio files and submit the predicted text to
Kaggle for scoring.

## Task

Build an ASR inference pipeline that:

1. Loads each test audio file from the competition data.
2. Produces a Thai text transcript for each audio file.
3. Writes a Kaggle submission CSV with one row per audio file.
4. Optimizes transcripts for minimum edit distance against the hidden ground
   truth.

The competition appears to be test/inference focused: the visible file listing
contains WAV audio under `audio_final/audio/`. No training labels or sample
submission file were visible through the API pages queried.

## Data

Visible data files are WAV audio clips:

```text
audio_final/audio/AU_015a21ca-7e23-4b70-93e6-da8a1bb8eaab.wav
audio_final/audio/AU_02164b42-47ac-4d45-aa3a-0288ede4380e.wav
audio_final/audio/AU_0225a06d-7949-49a6-84a0-0f4c2d9005aa.wav
...
```

The first Kaggle file page showed 200 WAV files and a next-page token, so the
full test set contains more than 200 audio files. The first listed files range
from roughly 0.23 MB to 4.12 MB. File creation timestamps on the visible page
are from 2026-06-05.

Recommended local layout after download:

```text
data/
  raw/
    audio_final/
      audio/
        AU_*.wav
  processed/
  submissions/
```

Download command:

```bash
kaggle competitions download -c individual-test-thai-call-center-asr -p data/raw
unzip data/raw/individual-test-thai-call-center-asr.zip -d data/raw
```

## Evaluation

Kaggle evaluates submissions using mean Levenshtein distance, described on the
competition evaluation page as Character/Word Error Rate:

```text
Error Rate = (S + D + I) / N
```

Where:

- `S` = substitutions
- `D` = deletions
- `I` = insertions
- `N` = total characters or words in the ground-truth reference

Lower scores are better. The Kaggle page specifically notes that exact Thai
characters and spacing are important.

Current visible leaderboard metadata from the API showed one baseline entry:

```text
rank: 1
team: baseline
score: 4.53538
```

## Submission

The exact sample submission file was not exposed in the queried file pages.
Prepare the submission with:

- one row per `AU_*.wav` file;
- an ID derived from the audio filename, usually the basename without `.wav`;
- one prediction column containing the Thai transcript.

Before making serious submissions, confirm the required column names from the
Kaggle UI or a sample submission if one appears after downloading all files.
A likely submission shape is:

```csv
id,transcript
AU_015a21ca-7e23-4b70-93e6-da8a1bb8eaab,สวัสดีค่ะ...
```

## Modeling Notes

Useful baseline approach:

1. Start with an open-source Thai-capable ASR model such as Whisper or a
   Thai-finetuned Wav2Vec2/Whisper model.
2. Normalize audio consistently before inference.
3. Preserve Thai script carefully; avoid destructive normalization that removes
   meaningful characters.
4. Tune decoding parameters on any allowed validation data.
5. Post-process spacing, repeated characters, punctuation, and filler tokens
   only when it improves Levenshtein distance.
6. Keep every external dataset and model source documented for the forum and
   final reproducibility.

Potential public resources allowed by the rules include Mozilla Common Voice
Thai and Gowajee, provided their use is disclosed according to the competition
rules.

## Open Items To Verify

- Exact submission column names.
- Exact number of audio files after full download.
- Whether any sample submission file appears in later file-listing pages or in
  the downloaded archive.
- Concrete start date and final submission deadline.
- Whether scoring is character-level, word-level, or a host-specific blend.
