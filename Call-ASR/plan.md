# Thai Call Center ASR Colab Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Google Colab-ready Thai call-center ASR solution that downloads and extracts Kaggle competition data inside Colab, audits the audio, runs Thai Whisper-family inference, normalizes transcripts, validates the CSV contract, and writes `/content/submission.csv` without submitting through any API.

**Architecture:** Keep competition logic in small Python modules under `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/` and keep the Colab-facing entrypoint in `/Users/temicide/Documents/5_domain_final/Call-ASR/notebooks/colab_submission.py`. The notebook script configures Kaggle credentials without printing them, downloads and extracts competition files into `/content/input/individual-test-thai-call-center-asr/`, resolves Colab paths first and local paths second, calls the same harness used in local tests, and writes validated candidate artifacts plus the final submission. Intermediate raw transcripts, normalized transcripts, and per-file JSONL logs are stored outside the source package so every run is resumable and auditable.

**Tech Stack:** Python 3.10+, pandas, numpy, torch, transformers, datasets, soundfile, librosa, jiwer, pytest, Google Colab runtime, Kaggle CLI/Python API for data download, ffmpeg available through Colab system packages or apt install.

---

## File Structure

- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/pyproject.toml`: package metadata, runtime dependencies, pytest configuration, console scripts.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/__init__.py`: package version export.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/paths.py`: Colab-first and local-fallback path resolution for extracted competition inputs and writable outputs.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/submission.py`: sample submission loading, audio match checks, row-order preservation, and CSV validation.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/audio.py`: audio discovery, prefix parsing, duration/sample-rate probing, and 16 kHz mono loading.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/audit_audio.py`: CLI that writes audio inventory CSV and decode-failure JSONL.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/normalize_text.py`: named Thai-safe normalization policies used by proxy scoring and submission generation.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/score_proxy.py`: CER/WER scoring helpers and CLI for external validation predictions.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/infer.py`: resumable one-model inference harness with Whisper-family loading, chunking controls, GPU/CPU selection, raw and normalized output columns, and JSONL logging.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/ensemble.py`: prefix routing, confidence selection, and conservative character-vote utilities over candidate transcript CSVs.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/proxy_data.py`: public validation dataset preparation hooks for Common Voice Thai, FLEURS Thai, and locally staged validation manifests.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/notebooks/colab_submission.py`: Colab notebook script entrypoint that installs/imports Kaggle tooling, handles credentials securely, downloads/extracts data to `/content/input/...`, runs or resumes inference, validates, and writes `/content/submission.csv`.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/fixtures/sample_submission.csv`: three-row fixture preserving the real schema.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/fixtures/audio/*.wav`: generated tiny mono WAV fixtures with prefixes `RSP`, `AU`, and `SDB`.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_paths.py`: path resolver coverage for Colab and local layouts.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_submission.py`: CSV schema, ordering, missing audio, and empty transcript validation coverage.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_audio.py`: audio metadata and 16 kHz loading coverage.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_normalize_text.py`: named normalization policy coverage with Thai, English, digits, spaces, punctuation, emoji, and fillers.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_score_proxy.py`: edit-distance metric coverage using known reference/prediction pairs.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_infer_resume.py`: resume behavior and log writing with a fake ASR backend.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_ensemble.py`: prefix route and confidence-selection coverage.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_colab_notebook_contract.py`: static checks that the notebook script handles credential setup, downloads/extracts data, writes `/content/submission.csv`, and never calls Kaggle submission APIs.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/data/submissions/.gitkeep`: local candidate submission directory.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/.gitkeep`: local inference and audit logs directory.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/data/proxy/.gitkeep`: local proxy validation manifests and predictions directory.
- Create `/Users/temicide/Documents/5_domain_final/Call-ASR/README.md`: exact local and Colab execution commands, artifact locations, credential handling, and manual submission boundary.
- Modify no files outside `/Users/temicide/Documents/5_domain_final/Call-ASR/`.

### Task 1: Project Scaffold

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/pyproject.toml`
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/__init__.py`
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/fixtures/sample_submission.csv`
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/conftest.py`
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/data/submissions/.gitkeep`
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/.gitkeep`
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/data/proxy/.gitkeep`
- Test: `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_project_scaffold.py`

- [ ] **Step 1: Create a failing scaffold test**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_project_scaffold.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_project_directories_exist():
    expected_dirs = [
        ROOT / "src" / "call_asr",
        ROOT / "tests" / "fixtures",
        ROOT / "data" / "submissions",
        ROOT / "data" / "runs",
        ROOT / "data" / "proxy",
    ]

    missing = [str(path) for path in expected_dirs if not path.is_dir()]

    assert missing == []


def test_package_exports_version():
    import call_asr

    assert call_asr.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the scaffold test and verify it fails**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_project_scaffold.py -v
```

Expected output contains:

```text
FAILED tests/test_project_scaffold.py::test_package_exports_version - ModuleNotFoundError: No module named 'call_asr'
```

- [ ] **Step 3: Add package metadata and package init**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "call-asr"
version = "0.1.0"
description = "Thai call-center ASR Colab notebook harness"
requires-python = ">=3.10"
dependencies = [
    "datasets>=2.20.0",
    "jiwer>=3.0.4",
    "librosa>=0.10.2.post1",
    "numpy>=1.26.4",
    "pandas>=2.2.2",
    "pytest>=8.2.2",
    "soundfile>=0.12.1",
    "torch>=2.3.1",
    "transformers>=4.41.2",
]

[project.scripts]
call-asr-audit = "call_asr.audit_audio:main"
call-asr-infer = "call_asr.infer:main"
call-asr-score = "call_asr.score_proxy:main"
call-asr-ensemble = "call_asr.ensemble:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-ra"
```

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/__init__.py`:

```python
__version__ = "0.1.0"
```

Create directories:

```bash
mkdir -p /Users/temicide/Documents/5_domain_final/Call-ASR/tests/fixtures/audio
mkdir -p /Users/temicide/Documents/5_domain_final/Call-ASR/data/submissions
mkdir -p /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs
mkdir -p /Users/temicide/Documents/5_domain_final/Call-ASR/data/proxy
touch /Users/temicide/Documents/5_domain_final/Call-ASR/data/submissions/.gitkeep
touch /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/.gitkeep
touch /Users/temicide/Documents/5_domain_final/Call-ASR/data/proxy/.gitkeep
```

- [ ] **Step 4: Add CSV and WAV fixtures**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/fixtures/sample_submission.csv`:

```csv
file_name,text
RSP_001_audio.wav,
AU_001_audio.wav,
SDB_001_audio.wav,
```

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/conftest.py`:

```python
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="session", autouse=True)
def write_audio_fixtures():
    fixture_dir = Path(__file__).resolve().parent / "fixtures" / "audio"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    timeline = np.linspace(0, 0.25, int(sample_rate * 0.25), endpoint=False)
    signal = 0.05 * np.sin(2 * np.pi * 440 * timeline)
    for file_name in ["RSP_001_audio.wav", "AU_001_audio.wav", "SDB_001_audio.wav"]:
        sf.write(fixture_dir / file_name, signal, sample_rate)
```

- [ ] **Step 5: Run the scaffold test and verify it passes**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_project_scaffold.py -v
```

Expected output contains:

```text
tests/test_project_scaffold.py::test_expected_project_directories_exist PASSED
tests/test_project_scaffold.py::test_package_exports_version PASSED
```

- [ ] **Step 6: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
git add pyproject.toml src/call_asr/__init__.py tests/test_project_scaffold.py tests/conftest.py tests/fixtures/sample_submission.csv data/submissions/.gitkeep data/runs/.gitkeep data/proxy/.gitkeep
git commit -m "chore: scaffold call asr package"
```

Expected output contains:

```text
[codex/
 chore: scaffold call asr package
```

### Task 2: Kaggle and Local Path Resolution

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/paths.py`
- Test: `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_paths.py`

- [ ] **Step 1: Write failing path resolver tests**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_paths.py`:

```python
from pathlib import Path

from call_asr.paths import CompetitionPaths, resolve_competition_paths


def test_resolve_local_paths_with_fixture_layout(tmp_path):
    local_root = tmp_path / "Call-ASR"
    competition_dir = local_root / "data" / "individual-test-thai-call-center-asr"
    audio_dir = competition_dir / "audio_final" / "audio"
    audio_dir.mkdir(parents=True)
    sample_submission = competition_dir / "sample_submission.csv"
    sample_submission.write_text("file_name,text\nRSP_001_audio.wav,\n", encoding="utf-8")

    paths = resolve_competition_paths(colab_input_root=tmp_path / "missing", project_root=local_root)

    assert paths == CompetitionPaths(
        input_dir=competition_dir,
        audio_dir=audio_dir,
        sample_submission=sample_submission,
        working_dir=local_root / "data" / "runs",
        submissions_dir=local_root / "data" / "submissions",
        is_colab=False,
    )


def test_resolve_colab_paths_before_local_paths(tmp_path):
    colab_input_root = tmp_path / "content" / "input"
    competition_dir = colab_input_root / "individual-test-thai-call-center-asr"
    audio_dir = competition_dir / "audio_final" / "audio"
    audio_dir.mkdir(parents=True)
    sample_submission = competition_dir / "sample_submission.csv"
    sample_submission.write_text("file_name,text\nRSP_001_audio.wav,\n", encoding="utf-8")
    local_root = tmp_path / "Call-ASR"
    (local_root / "data" / "individual-test-thai-call-center-asr" / "audio_final" / "audio").mkdir(parents=True)
    (local_root / "data" / "individual-test-thai-call-center-asr" / "sample_submission.csv").write_text(
        "file_name,text\nSDB_001_audio.wav,\n", encoding="utf-8"
    )

    paths = resolve_competition_paths(colab_input_root=colab_input_root, project_root=local_root)

    assert paths.input_dir == competition_dir
    assert paths.audio_dir == audio_dir
    assert paths.sample_submission == sample_submission
    assert paths.working_dir == Path("/content/working")
    assert paths.submissions_dir == Path("/content/working")
    assert paths.is_colab is True
```

- [ ] **Step 2: Run path tests and verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_paths.py -v
```

Expected output contains:

```text
FAILED tests/test_paths.py - ModuleNotFoundError: No module named 'call_asr.paths'
```

- [ ] **Step 3: Implement the path resolver**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/paths.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


COMPETITION_SLUG = "individual-test-thai-call-center-asr"


@dataclass(frozen=True)
class CompetitionPaths:
    input_dir: Path
    audio_dir: Path
    sample_submission: Path
    working_dir: Path
    submissions_dir: Path
    is_colab: bool


def _has_competition_files(input_dir: Path) -> bool:
    return (
        (input_dir / "sample_submission.csv").is_file()
        and (input_dir / "audio_final" / "audio").is_dir()
    )


def resolve_competition_paths(
    colab_input_root: Path = Path("/content/input"),
    project_root: Path | None = None,
) -> CompetitionPaths:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    colab_competition_dir = colab_input_root / COMPETITION_SLUG
    local_competition_dir = project_root / "data" / COMPETITION_SLUG

    if _has_competition_files(colab_competition_dir):
        input_dir = colab_competition_dir
        working_dir = Path("/content/working")
        submissions_dir = Path("/content/working")
        is_colab = True
    elif _has_competition_files(local_competition_dir):
        input_dir = local_competition_dir
        working_dir = project_root / "data" / "runs"
        submissions_dir = project_root / "data" / "submissions"
        is_colab = False
    else:
        raise FileNotFoundError(
            "Competition files not found. Expected sample_submission.csv and "
            f"audio_final/audio under {colab_competition_dir} or {local_competition_dir}."
        )

    return CompetitionPaths(
        input_dir=input_dir,
        audio_dir=input_dir / "audio_final" / "audio",
        sample_submission=input_dir / "sample_submission.csv",
        working_dir=working_dir,
        submissions_dir=submissions_dir,
        is_colab=is_colab,
    )
```

- [ ] **Step 4: Run path tests and verify they pass**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_paths.py -v
```

Expected output contains:

```text
tests/test_paths.py::test_resolve_local_paths_with_fixture_layout PASSED
tests/test_paths.py::test_resolve_kaggle_paths_before_local_paths PASSED
```

- [ ] **Step 5: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
git add src/call_asr/paths.py tests/test_paths.py
git commit -m "feat: resolve colab and local competition paths"
```

Expected output contains:

```text
[codex/
 feat: resolve colab and local competition paths
```

### Task 3: Submission Contract Validation

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/submission.py`
- Test: `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_submission.py`

- [ ] **Step 1: Write failing submission tests**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_submission.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from call_asr.submission import (
    SubmissionValidationError,
    load_sample_submission,
    validate_audio_coverage,
    validate_submission_frame,
    write_submission_csv,
)


def test_load_sample_submission_preserves_schema_and_order(tmp_path):
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("file_name,text\nRSP_002_audio.wav,\nRSP_001_audio.wav,\n", encoding="utf-8")

    df = load_sample_submission(sample)

    assert list(df.columns) == ["file_name", "text"]
    assert df["file_name"].tolist() == ["RSP_002_audio.wav", "RSP_001_audio.wav"]


def test_validate_audio_coverage_reports_missing_file(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "RSP_001_audio.wav").write_bytes(b"fake")
    df = pd.DataFrame({"file_name": ["RSP_001_audio.wav", "SDB_001_audio.wav"], "text": ["", ""]})

    with pytest.raises(SubmissionValidationError, match="Missing audio files: SDB_001_audio.wav"):
        validate_audio_coverage(df, audio_dir)


def test_validate_submission_frame_rejects_empty_non_silence_text():
    df = pd.DataFrame({"file_name": ["RSP_001_audio.wav"], "text": [""]})

    with pytest.raises(SubmissionValidationError, match="Empty transcript for RSP_001_audio.wav"):
        validate_submission_frame(df, allow_empty_files=set())


def test_write_submission_csv_preserves_sample_order(tmp_path):
    output = tmp_path / "submission.csv"
    sample_df = pd.DataFrame({"file_name": ["RSP_002_audio.wav", "RSP_001_audio.wav"], "text": ["", ""]})
    predictions = pd.DataFrame(
        {
            "file_name": ["RSP_001_audio.wav", "RSP_002_audio.wav"],
            "normalized_text": ["หนึ่ง", "สอง"],
        }
    )

    write_submission_csv(sample_df, predictions, output)

    written = pd.read_csv(output)
    assert written.to_dict("records") == [
        {"file_name": "RSP_002_audio.wav", "text": "สอง"},
        {"file_name": "RSP_001_audio.wav", "text": "หนึ่ง"},
    ]
```

- [ ] **Step 2: Run submission tests and verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_submission.py -v
```

Expected output contains:

```text
FAILED tests/test_submission.py - ModuleNotFoundError: No module named 'call_asr.submission'
```

- [ ] **Step 3: Implement submission validation**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/submission.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd


class SubmissionValidationError(ValueError):
    pass


def load_sample_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"file_name": "string", "text": "string"}, keep_default_na=False)
    if list(df.columns) != ["file_name", "text"]:
        raise SubmissionValidationError(
            f"Expected sample submission columns ['file_name', 'text'], got {list(df.columns)}"
        )
    if df["file_name"].duplicated().any():
        duplicates = df.loc[df["file_name"].duplicated(), "file_name"].tolist()
        raise SubmissionValidationError(f"Duplicate file_name values: {', '.join(duplicates[:10])}")
    return df


def validate_audio_coverage(sample_df: pd.DataFrame, audio_dir: Path) -> None:
    missing = [name for name in sample_df["file_name"].tolist() if not (audio_dir / name).is_file()]
    if missing:
        raise SubmissionValidationError(f"Missing audio files: {', '.join(missing[:20])}")


def validate_submission_frame(submission_df: pd.DataFrame, allow_empty_files: set[str] | None = None) -> None:
    allow_empty_files = allow_empty_files or set()
    if list(submission_df.columns) != ["file_name", "text"]:
        raise SubmissionValidationError(
            f"Expected submission columns ['file_name', 'text'], got {list(submission_df.columns)}"
        )
    if submission_df["file_name"].duplicated().any():
        duplicates = submission_df.loc[submission_df["file_name"].duplicated(), "file_name"].tolist()
        raise SubmissionValidationError(f"Duplicate submission rows: {', '.join(duplicates[:10])}")
    for row in submission_df.itertuples(index=False):
        text = "" if pd.isna(row.text) else str(row.text)
        if text == "" and row.file_name not in allow_empty_files:
            raise SubmissionValidationError(f"Empty transcript for {row.file_name}")


def write_submission_csv(
    sample_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_path: Path,
    allow_empty_files: set[str] | None = None,
) -> pd.DataFrame:
    required_prediction_columns = {"file_name", "normalized_text"}
    missing_columns = required_prediction_columns - set(predictions_df.columns)
    if missing_columns:
        raise SubmissionValidationError(f"Prediction columns missing: {', '.join(sorted(missing_columns))}")

    merged = sample_df[["file_name"]].merge(
        predictions_df[["file_name", "normalized_text"]],
        on="file_name",
        how="left",
        validate="one_to_one",
    )
    if merged["normalized_text"].isna().any():
        missing = merged.loc[merged["normalized_text"].isna(), "file_name"].tolist()
        raise SubmissionValidationError(f"Missing predictions: {', '.join(missing[:20])}")

    submission_df = merged.rename(columns={"normalized_text": "text"})[["file_name", "text"]]
    submission_df["text"] = submission_df["text"].fillna("").astype(str)
    validate_submission_frame(submission_df, allow_empty_files=allow_empty_files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return submission_df
```

- [ ] **Step 4: Run submission tests and verify they pass**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_submission.py -v
```

Expected output contains:

```text
tests/test_submission.py::test_load_sample_submission_preserves_schema_and_order PASSED
tests/test_submission.py::test_validate_audio_coverage_reports_missing_file PASSED
tests/test_submission.py::test_validate_submission_frame_rejects_empty_non_silence_text PASSED
tests/test_submission.py::test_write_submission_csv_preserves_sample_order PASSED
```

- [ ] **Step 5: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
git add src/call_asr/submission.py tests/test_submission.py
git commit -m "feat: validate kaggle submission contract"
```

Expected output contains:

```text
[codex/
 feat: validate kaggle submission contract
```

### Task 4: Audio Metadata and Loading

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/audio.py`
- Test: `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_audio.py`

- [ ] **Step 1: Write failing audio tests**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_audio.py`:

```python
from pathlib import Path

import numpy as np
import soundfile as sf

from call_asr.audio import audio_prefix, load_audio_16khz, probe_audio_file


def test_audio_prefix_returns_text_before_first_underscore():
    assert audio_prefix("AU_001_audio.wav") == "AU"
    assert audio_prefix("RSP_101_audio.wav") == "RSP"


def test_probe_audio_file_reports_duration_sample_rate_and_channels(tmp_path):
    wav_path = tmp_path / "SDB_001_audio.wav"
    samples = np.zeros(8000, dtype=np.float32)
    sf.write(wav_path, samples, 8000)

    metadata = probe_audio_file(wav_path)

    assert metadata.file_name == "SDB_001_audio.wav"
    assert metadata.prefix == "SDB"
    assert metadata.sample_rate == 8000
    assert metadata.channels == 1
    assert round(metadata.duration_seconds, 2) == 1.0
    assert metadata.decode_ok is True
    assert metadata.error == ""


def test_load_audio_16khz_resamples_and_returns_mono(tmp_path):
    wav_path = tmp_path / "AU_001_audio.wav"
    samples = np.zeros((4000, 2), dtype=np.float32)
    samples[:, 0] = 0.1
    samples[:, 1] = -0.1
    sf.write(wav_path, samples, 8000)

    audio, sample_rate = load_audio_16khz(wav_path)

    assert sample_rate == 16000
    assert audio.ndim == 1
    assert 7900 <= len(audio) <= 8100
```

- [ ] **Step 2: Run audio tests and verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_audio.py -v
```

Expected output contains:

```text
FAILED tests/test_audio.py - ModuleNotFoundError: No module named 'call_asr.audio'
```

- [ ] **Step 3: Implement audio helpers**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/audio.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AudioMetadata:
    file_name: str
    prefix: str
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    decode_ok: bool
    error: str


def audio_prefix(file_name: str) -> str:
    return file_name.split("_", 1)[0]


def probe_audio_file(path: Path) -> AudioMetadata:
    try:
        info = sf.info(path)
        channels = int(info.channels)
        sample_rate = int(info.samplerate)
        frames = int(info.frames)
        duration_seconds = float(frames / sample_rate) if sample_rate else 0.0
        return AudioMetadata(
            file_name=path.name,
            prefix=audio_prefix(path.name),
            sample_rate=sample_rate,
            channels=channels,
            frames=frames,
            duration_seconds=duration_seconds,
            decode_ok=True,
            error="",
        )
    except Exception as exc:
        return AudioMetadata(
            file_name=path.name,
            prefix=audio_prefix(path.name),
            sample_rate=0,
            channels=0,
            frames=0,
            duration_seconds=0.0,
            decode_ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def load_audio_16khz(path: Path) -> tuple[np.ndarray, int]:
    audio, source_sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if source_sample_rate != TARGET_SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=source_sample_rate, target_sr=TARGET_SAMPLE_RATE)
    return np.asarray(mono, dtype=np.float32), TARGET_SAMPLE_RATE
```

- [ ] **Step 4: Run audio tests and verify they pass**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_audio.py -v
```

Expected output contains:

```text
tests/test_audio.py::test_audio_prefix_returns_text_before_first_underscore PASSED
tests/test_audio.py::test_probe_audio_file_reports_duration_sample_rate_and_channels PASSED
tests/test_audio.py::test_load_audio_16khz_resamples_and_returns_mono PASSED
```

- [ ] **Step 5: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
git add src/call_asr/audio.py tests/test_audio.py
git commit -m "feat: add audio probing and loading"
```

Expected output contains:

```text
[codex/
 feat: add audio probing and loading
```

### Task 5: Full Audio Audit CLI

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/audit_audio.py`
- Test: `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_audit_audio.py`

- [ ] **Step 1: Write failing audit tests**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_audit_audio.py`:

```python
import json

import pandas as pd

from call_asr.audit_audio import audit_audio_directory


def test_audit_audio_directory_writes_inventory_and_failures(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    bad_file = audio_dir / "RSP_bad_audio.wav"
    bad_file.write_text("not a wav", encoding="utf-8")
    output_csv = tmp_path / "audio_inventory.csv"
    failures_jsonl = tmp_path / "decode_failures.jsonl"

    inventory = audit_audio_directory(audio_dir, output_csv, failures_jsonl)

    assert inventory.loc[0, "file_name"] == "RSP_bad_audio.wav"
    assert inventory.loc[0, "decode_ok"] is False or inventory.loc[0, "decode_ok"] == False
    assert output_csv.is_file()
    assert failures_jsonl.is_file()
    failure = json.loads(failures_jsonl.read_text(encoding="utf-8").strip())
    assert failure["file_name"] == "RSP_bad_audio.wav"
    assert "error" in failure


def test_audit_audio_directory_sorts_by_file_name(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for name in ["SDB_002_audio.wav", "SDB_001_audio.wav"]:
        (audio_dir / name).write_text("not a wav", encoding="utf-8")
    output_csv = tmp_path / "audio_inventory.csv"
    failures_jsonl = tmp_path / "decode_failures.jsonl"

    audit_audio_directory(audio_dir, output_csv, failures_jsonl)

    written = pd.read_csv(output_csv)
    assert written["file_name"].tolist() == ["SDB_001_audio.wav", "SDB_002_audio.wav"]
```

- [ ] **Step 2: Run audit tests and verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_audit_audio.py -v
```

Expected output contains:

```text
FAILED tests/test_audit_audio.py - ModuleNotFoundError: No module named 'call_asr.audit_audio'
```

- [ ] **Step 3: Implement the audit CLI**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/audit_audio.py`:

```python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from call_asr.audio import probe_audio_file
from call_asr.paths import resolve_competition_paths


def audit_audio_directory(audio_dir: Path, output_csv: Path, failures_jsonl: Path) -> pd.DataFrame:
    rows = [asdict(probe_audio_file(path)) for path in sorted(audio_dir.glob("*.wav"))]
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        inventory = pd.DataFrame(
            columns=[
                "file_name",
                "prefix",
                "sample_rate",
                "channels",
                "frames",
                "duration_seconds",
                "decode_ok",
                "error",
            ]
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    failures_jsonl.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_csv, index=False)
    failures = inventory[inventory["decode_ok"] == False]
    with failures_jsonl.open("w", encoding="utf-8") as handle:
        for row in failures.to_dict("records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return inventory


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Thai call-center WAV files.")
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--failures-jsonl", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = resolve_competition_paths()
    audio_dir = args.audio_dir or paths.audio_dir
    output_csv = args.output_csv or (paths.working_dir / "audio_inventory.csv")
    failures_jsonl = args.failures_jsonl or (paths.working_dir / "decode_failures.jsonl")
    inventory = audit_audio_directory(audio_dir, output_csv, failures_jsonl)
    prefix_counts = inventory["prefix"].value_counts().sort_index().to_dict()
    print(f"Audited {len(inventory)} WAV files")
    print(f"Decode failures: {int((inventory['decode_ok'] == False).sum())}")
    print(f"Prefix counts: {prefix_counts}")
    print(f"Wrote inventory: {output_csv}")
    print(f"Wrote failures: {failures_jsonl}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run audit tests and verify they pass**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_audit_audio.py -v
```

Expected output contains:

```text
tests/test_audit_audio.py::test_audit_audio_directory_writes_inventory_and_failures PASSED
tests/test_audit_audio.py::test_audit_audio_directory_sorts_by_file_name PASSED
```

- [ ] **Step 5: Run the audit on the real local data**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m call_asr.audit_audio \
  --audio-dir /Users/temicide/Documents/5_domain_final/Call-ASR/data/individual-test-thai-call-center-asr/audio_final/audio \
  --output-csv /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/audio_inventory.csv \
  --failures-jsonl /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/decode_failures.jsonl
```

Expected output contains:

```text
Audited 6261 WAV files
Decode failures: 0
Prefix counts: {'AU': 400, 'BCH': 240, 'FD': 11, 'INT': 1080, 'RSP': 720, 'SDB': 3330, 'TT': 480}
Wrote inventory: /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/audio_inventory.csv
Wrote failures: /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/decode_failures.jsonl
```

- [ ] **Step 6: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
git add src/call_asr/audit_audio.py tests/test_audit_audio.py data/runs/.gitkeep
git commit -m "feat: audit competition audio files"
```

Expected output contains:

```text
[codex/
 feat: audit competition audio files
```

### Task 6: Thai-Safe Text Normalization

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/normalize_text.py`
- Test: `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_normalize_text.py`

- [ ] **Step 1: Write failing normalization tests**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_normalize_text.py`:

```python
import pytest

from call_asr.normalize_text import normalize_text


@pytest.mark.parametrize(
    ("policy", "raw", "expected"),
    [
        ("raw", "  สวัสดีค่ะ!!!  ", "สวัสดีค่ะ!!!"),
        ("single_space", "สวัสดี   ค่ะ\nโทร  123", "สวัสดี ค่ะ โทร 123"),
        ("no_spaces", "สวัสดี   ค่ะ โทร 123", "สวัสดีค่ะโทร123"),
        ("thai_chars_only_light", "ค่ะ! โทร ABC 123 😊", "ค่ะ โทร ABC 123"),
        ("remove_fillers", "เอ่อ สวัสดี อืม ค่ะ", "สวัสดี ค่ะ"),
    ],
)
def test_normalize_text_named_policies(policy, raw, expected):
    assert normalize_text(raw, policy) == expected


def test_normalize_text_rejects_unknown_policy():
    with pytest.raises(ValueError, match="Unknown normalization policy: missing"):
        normalize_text("สวัสดี", "missing")
```

- [ ] **Step 2: Run normalization tests and verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_normalize_text.py -v
```

Expected output contains:

```text
FAILED tests/test_normalize_text.py - ModuleNotFoundError: No module named 'call_asr.normalize_text'
```

- [ ] **Step 3: Implement normalization policies**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/normalize_text.py`:

```python
from __future__ import annotations

import re
import unicodedata


FILLERS = ("เอ่อ", "อืม", "อ่า", "อะ", "แบบว่า")
POLICIES = {"raw", "single_space", "no_spaces", "thai_chars_only_light", "remove_fillers"}


def _strip_control_chars(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch.isspace())


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _light_allowed_chars(text: str) -> str:
    kept = []
    for ch in text:
        codepoint = ord(ch)
        is_thai = 0x0E00 <= codepoint <= 0x0E7F
        is_ascii_letter = "A" <= ch <= "Z" or "a" <= ch <= "z"
        is_digit = ch.isdigit()
        if is_thai or is_ascii_letter or is_digit or ch.isspace():
            kept.append(ch)
        else:
            kept.append(" ")
    return _collapse_spaces("".join(kept))


def normalize_text(text: str, policy: str) -> str:
    if policy not in POLICIES:
        raise ValueError(f"Unknown normalization policy: {policy}")

    cleaned = _strip_control_chars(str(text)).strip()
    if policy == "raw":
        return cleaned
    if policy == "single_space":
        return _collapse_spaces(cleaned)
    if policy == "no_spaces":
        return re.sub(r"\s+", "", cleaned)
    if policy == "thai_chars_only_light":
        return _light_allowed_chars(cleaned)
    if policy == "remove_fillers":
        without_fillers = cleaned
        for filler in FILLERS:
            without_fillers = re.sub(rf"(^|\s){re.escape(filler)}(?=\s|$)", " ", without_fillers)
        return _collapse_spaces(without_fillers)

    raise ValueError(f"Unknown normalization policy: {policy}")
```

- [ ] **Step 4: Run normalization tests and verify they pass**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_normalize_text.py -v
```

Expected output contains:

```text
tests/test_normalize_text.py::test_normalize_text_named_policies[raw-  สวัสดีค่ะ!!!  -สวัสดีค่ะ!!!] PASSED
tests/test_normalize_text.py::test_normalize_text_rejects_unknown_policy PASSED
```

- [ ] **Step 5: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
git add src/call_asr/normalize_text.py tests/test_normalize_text.py
git commit -m "feat: add thai transcript normalization policies"
```

Expected output contains:

```text
[codex/
 feat: add thai transcript normalization policies
```

### Task 7: Proxy Scoring Metrics

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/score_proxy.py`
- Test: `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_score_proxy.py`

- [ ] **Step 1: Write failing scoring tests**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_score_proxy.py`:

```python
import pandas as pd

from call_asr.score_proxy import character_error_rate, score_predictions


def test_character_error_rate_uses_edit_distance_over_reference_length():
    assert character_error_rate("abc", "axbc") == 1 / 3
    assert character_error_rate("สวัสดี", "สวัสดี") == 0.0


def test_score_predictions_returns_overall_and_by_prefix():
    reference = pd.DataFrame(
        {
            "file_name": ["RSP_001_audio.wav", "AU_001_audio.wav"],
            "text": ["abc", "สวัสดี"],
        }
    )
    predictions = pd.DataFrame(
        {
            "file_name": ["AU_001_audio.wav", "RSP_001_audio.wav"],
            "normalized_text": ["สวัสดี", "axbc"],
        }
    )

    result = score_predictions(reference, predictions)

    assert round(result["overall_cer"], 4) == round(1 / 9, 4)
    assert result["by_prefix"]["AU"]["count"] == 1
    assert result["by_prefix"]["AU"]["cer"] == 0.0
    assert result["by_prefix"]["RSP"]["count"] == 1
    assert round(result["by_prefix"]["RSP"]["cer"], 4) == round(1 / 3, 4)
```

- [ ] **Step 2: Run scoring tests and verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_score_proxy.py -v
```

Expected output contains:

```text
FAILED tests/test_score_proxy.py - ModuleNotFoundError: No module named 'call_asr.score_proxy'
```

- [ ] **Step 3: Implement proxy scoring**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/score_proxy.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from call_asr.audio import audio_prefix


def _edit_distance(reference: str, prediction: str) -> int:
    previous = list(range(len(prediction) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, pred_char in enumerate(prediction, start=1):
            substitution_cost = 0 if ref_char == pred_char else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, prediction: str) -> float:
    reference = str(reference)
    prediction = str(prediction)
    if len(reference) == 0:
        return 0.0 if prediction == "" else 1.0
    return _edit_distance(reference, prediction) / len(reference)


def score_predictions(reference_df: pd.DataFrame, predictions_df: pd.DataFrame) -> dict:
    merged = reference_df[["file_name", "text"]].merge(
        predictions_df[["file_name", "normalized_text"]],
        on="file_name",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(reference_df):
        missing = sorted(set(reference_df["file_name"]) - set(merged["file_name"]))
        raise ValueError(f"Missing prediction rows: {', '.join(missing[:20])}")

    total_distance = 0
    total_reference_chars = 0
    by_prefix: dict[str, dict[str, float | int]] = {}
    for row in merged.itertuples(index=False):
        reference = str(row.text)
        prediction = str(row.normalized_text)
        distance = _edit_distance(reference, prediction)
        ref_len = len(reference)
        prefix = audio_prefix(row.file_name)
        total_distance += distance
        total_reference_chars += ref_len
        stats = by_prefix.setdefault(prefix, {"distance": 0, "reference_chars": 0, "count": 0, "cer": 0.0})
        stats["distance"] += distance
        stats["reference_chars"] += ref_len
        stats["count"] += 1

    for stats in by_prefix.values():
        stats["cer"] = 0.0 if stats["reference_chars"] == 0 else stats["distance"] / stats["reference_chars"]

    return {
        "overall_cer": 0.0 if total_reference_chars == 0 else total_distance / total_reference_chars,
        "count": int(len(merged)),
        "by_prefix": by_prefix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score proxy ASR predictions.")
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    reference_df = pd.read_csv(args.reference_csv, keep_default_na=False)
    predictions_df = pd.read_csv(args.predictions_csv, keep_default_na=False)
    scores = score_predictions(reference_df, predictions_df)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run scoring tests and verify they pass**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_score_proxy.py -v
```

Expected output contains:

```text
tests/test_score_proxy.py::test_character_error_rate_uses_edit_distance_over_reference_length PASSED
tests/test_score_proxy.py::test_score_predictions_returns_overall_and_by_prefix PASSED
```

- [ ] **Step 5: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
git add src/call_asr/score_proxy.py tests/test_score_proxy.py
git commit -m "feat: add proxy cer scoring"
```

Expected output contains:

```text
[codex/
 feat: add proxy cer scoring
```

### Task 8: Resumable Whisper Inference Harness

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/infer.py`
- Test: `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_infer_resume.py`

- [ ] **Step 1: Write failing inference tests with a fake backend**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_infer_resume.py`:

```python
import json

import pandas as pd

from call_asr.infer import AsrResult, FakeAsrBackend, run_inference


def test_run_inference_writes_predictions_and_jsonl_log(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "RSP_001_audio.wav").write_bytes(b"fake")
    sample_df = pd.DataFrame({"file_name": ["RSP_001_audio.wav"], "text": [""]})
    output_csv = tmp_path / "predictions.csv"
    log_jsonl = tmp_path / "run.jsonl"
    backend = FakeAsrBackend({"RSP_001_audio.wav": AsrResult(text=" สวัสดีค่ะ ", avg_logprob=-0.1, compression_ratio=1.2, no_speech_prob=0.01)})

    predictions = run_inference(
        sample_df=sample_df,
        audio_dir=audio_dir,
        output_csv=output_csv,
        log_jsonl=log_jsonl,
        backend=backend,
        normalization_policy="single_space",
        resume=True,
    )

    assert predictions.to_dict("records") == [
        {
            "file_name": "RSP_001_audio.wav",
            "raw_text": " สวัสดีค่ะ ",
            "normalized_text": "สวัสดีค่ะ",
            "model_name": "fake",
            "avg_logprob": -0.1,
            "compression_ratio": 1.2,
            "no_speech_prob": 0.01,
            "error": "",
        }
    ]
    log_row = json.loads(log_jsonl.read_text(encoding="utf-8").strip())
    assert log_row["file_name"] == "RSP_001_audio.wav"
    assert log_row["normalized_text"] == "สวัสดีค่ะ"


def test_run_inference_resumes_existing_rows(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "RSP_001_audio.wav").write_bytes(b"fake")
    (audio_dir / "SDB_001_audio.wav").write_bytes(b"fake")
    sample_df = pd.DataFrame({"file_name": ["RSP_001_audio.wav", "SDB_001_audio.wav"], "text": ["", ""]})
    output_csv = tmp_path / "predictions.csv"
    log_jsonl = tmp_path / "run.jsonl"
    pd.DataFrame(
        [
            {
                "file_name": "RSP_001_audio.wav",
                "raw_text": "เดิม",
                "normalized_text": "เดิม",
                "model_name": "fake",
                "avg_logprob": -0.2,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
                "error": "",
            }
        ]
    ).to_csv(output_csv, index=False)
    backend = FakeAsrBackend({"SDB_001_audio.wav": AsrResult(text="ใหม่", avg_logprob=-0.3, compression_ratio=1.1, no_speech_prob=0.02)})

    predictions = run_inference(
        sample_df=sample_df,
        audio_dir=audio_dir,
        output_csv=output_csv,
        log_jsonl=log_jsonl,
        backend=backend,
        normalization_policy="raw",
        resume=True,
    )

    assert predictions["file_name"].tolist() == ["RSP_001_audio.wav", "SDB_001_audio.wav"]
    assert backend.calls == ["SDB_001_audio.wav"]
```

- [ ] **Step 2: Run inference tests and verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_infer_resume.py -v
```

Expected output contains:

```text
FAILED tests/test_infer_resume.py - ModuleNotFoundError: No module named 'call_asr.infer'
```

- [ ] **Step 3: Implement fake and Whisper backends plus resumable inference**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/infer.py`:

```python
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from call_asr.normalize_text import normalize_text
from call_asr.paths import resolve_competition_paths
from call_asr.submission import load_sample_submission, validate_audio_coverage


PREDICTION_COLUMNS = [
    "file_name",
    "raw_text",
    "normalized_text",
    "model_name",
    "avg_logprob",
    "compression_ratio",
    "no_speech_prob",
    "error",
]


@dataclass(frozen=True)
class AsrResult:
    text: str
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class AsrBackend(Protocol):
    model_name: str

    def transcribe(self, audio_path: Path) -> AsrResult:
        ...


class FakeAsrBackend:
    model_name = "fake"

    def __init__(self, outputs: dict[str, AsrResult]):
        self.outputs = outputs
        self.calls: list[str] = []

    def transcribe(self, audio_path: Path) -> AsrResult:
        self.calls.append(audio_path.name)
        return self.outputs[audio_path.name]


class WhisperPipelineBackend:
    def __init__(
        self,
        model_name: str,
        chunk_length_s: int,
        batch_size: int,
        condition_on_previous_text: bool,
        device: str,
    ):
        self.model_name = model_name
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        device_index = 0 if device == "cuda" else -1
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        processor = AutoProcessor.from_pretrained(model_name)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            device=device_index,
            return_timestamps=False,
        )
        self.generate_kwargs = {
            "language": "thai",
            "task": "transcribe",
            "condition_on_prev_tokens": condition_on_previous_text,
        }

    def transcribe(self, audio_path: Path) -> AsrResult:
        output = self.pipe(str(audio_path), generate_kwargs=self.generate_kwargs)
        text = str(output.get("text", ""))
        return AsrResult(text=text, avg_logprob=0.0, compression_ratio=0.0, no_speech_prob=0.0)


def _load_existing(output_csv: Path) -> pd.DataFrame:
    if output_csv.is_file():
        return pd.read_csv(output_csv, keep_default_na=False)
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def _write_log(log_jsonl: Path, row: dict) -> None:
    log_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with log_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_inference(
    sample_df: pd.DataFrame,
    audio_dir: Path,
    output_csv: Path,
    log_jsonl: Path,
    backend: AsrBackend,
    normalization_policy: str,
    resume: bool,
) -> pd.DataFrame:
    existing = _load_existing(output_csv) if resume else pd.DataFrame(columns=PREDICTION_COLUMNS)
    completed = set(existing["file_name"].tolist()) if not existing.empty else set()
    rows = existing.to_dict("records")

    for file_name in sample_df["file_name"].tolist():
        if file_name in completed:
            continue
        started = time.time()
        audio_path = audio_dir / file_name
        try:
            result = backend.transcribe(audio_path)
            raw_text = result.text
            normalized_text = normalize_text(raw_text, normalization_policy)
            row = {
                "file_name": file_name,
                "raw_text": raw_text,
                "normalized_text": normalized_text,
                "model_name": backend.model_name,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
                "error": "",
            }
        except Exception as exc:
            row = {
                "file_name": file_name,
                "raw_text": "",
                "normalized_text": "",
                "model_name": backend.model_name,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        log_row = dict(row)
        log_row["runtime_seconds"] = round(time.time() - started, 4)
        _write_log(log_jsonl, log_row)
        rows.append(row)
        pd.DataFrame(rows, columns=PREDICTION_COLUMNS).to_csv(output_csv, index=False)

    output = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    order = {file_name: position for position, file_name in enumerate(sample_df["file_name"].tolist())}
    output["_order"] = output["file_name"].map(order)
    output = output.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    output.to_csv(output_csv, index=False)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Thai ASR inference.")
    parser.add_argument("--model-name", default="typhoon-ai/typhoon-whisper-large-v3")
    parser.add_argument("--normalization-policy", default="single_space")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--log-jsonl", type=Path, default=None)
    parser.add_argument("--chunk-length-s", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--condition-on-previous-text", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = resolve_competition_paths()
    sample_df = load_sample_submission(paths.sample_submission)
    validate_audio_coverage(sample_df, paths.audio_dir)
    safe_model_name = args.model_name.replace("/", "__")
    output_csv = args.output_csv or (paths.working_dir / f"predictions_{safe_model_name}_{args.normalization_policy}.csv")
    log_jsonl = args.log_jsonl or (paths.working_dir / f"run_{safe_model_name}_{args.normalization_policy}.jsonl")
    backend = WhisperPipelineBackend(
        model_name=args.model_name,
        chunk_length_s=args.chunk_length_s,
        batch_size=args.batch_size,
        condition_on_previous_text=args.condition_on_previous_text,
        device=args.device,
    )
    predictions = run_inference(
        sample_df=sample_df,
        audio_dir=paths.audio_dir,
        output_csv=output_csv,
        log_jsonl=log_jsonl,
        backend=backend,
        normalization_policy=args.normalization_policy,
        resume=not args.no_resume,
    )
    errors = int((predictions["error"] != "").sum())
    print(f"Wrote predictions: {output_csv}")
    print(f"Wrote log: {log_jsonl}")
    print(f"Rows: {len(predictions)}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run inference tests and verify they pass**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_infer_resume.py -v
```

Expected output contains:

```text
tests/test_infer_resume.py::test_run_inference_writes_predictions_and_jsonl_log PASSED
tests/test_infer_resume.py::test_run_inference_resumes_existing_rows PASSED
```

- [ ] **Step 5: Run a 50-file local smoke command with the primary model**

Run this only on a GPU machine with enough memory for Whisper Large v3:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 - <<'PY'
from pathlib import Path
import pandas as pd
from call_asr.infer import WhisperPipelineBackend, run_inference
from call_asr.paths import resolve_competition_paths
from call_asr.submission import load_sample_submission

paths = resolve_competition_paths(project_root=Path('/Users/temicide/Documents/5_domain_final/Call-ASR'))
sample_df = load_sample_submission(paths.sample_submission).head(50)
backend = WhisperPipelineBackend(
    model_name='typhoon-ai/typhoon-whisper-large-v3',
    chunk_length_s=30,
    batch_size=4,
    condition_on_previous_text=False,
    device='cuda',
)
predictions = run_inference(
    sample_df=sample_df,
    audio_dir=paths.audio_dir,
    output_csv=paths.working_dir / 'smoke_typhoon_50_single_space.csv',
    log_jsonl=paths.working_dir / 'smoke_typhoon_50_single_space.jsonl',
    backend=backend,
    normalization_policy='single_space',
    resume=True,
)
print(predictions[['file_name', 'normalized_text', 'error']].head().to_string(index=False))
print(f'rows={len(predictions)} errors={(predictions["error"] != "").sum()}')
PY
```

Expected output contains:

```text
rows=50 errors=0
```

- [ ] **Step 6: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
git add src/call_asr/infer.py tests/test_infer_resume.py
git commit -m "feat: add resumable whisper inference"
```

Expected output contains:

```text
[codex/
 feat: add resumable whisper inference
```

### Task 9: Proxy Validation Dataset Hooks

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/proxy_data.py`
- Test: `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_proxy_data.py`

- [ ] **Step 1: Write failing proxy dataset tests**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_proxy_data.py`:

```python
import pandas as pd

from call_asr.proxy_data import degrade_manifest_for_call_center, validate_proxy_manifest


def test_validate_proxy_manifest_accepts_required_columns(tmp_path):
    manifest = tmp_path / "proxy.csv"
    pd.DataFrame(
        {
            "file_name": ["cv_th_001.wav"],
            "audio_path": ["/tmp/cv_th_001.wav"],
            "text": ["สวัสดีค่ะ"],
            "source": ["common_voice_th"],
            "split": ["validation"],
        }
    ).to_csv(manifest, index=False)

    df = validate_proxy_manifest(manifest)

    assert df["source"].tolist() == ["common_voice_th"]


def test_degrade_manifest_for_call_center_adds_transform_columns():
    df = pd.DataFrame(
        {
            "file_name": ["cv_th_001.wav"],
            "audio_path": ["/tmp/cv_th_001.wav"],
            "text": ["สวัสดีค่ะ"],
            "source": ["common_voice_th"],
            "split": ["validation"],
        }
    )

    degraded = degrade_manifest_for_call_center(df)

    assert degraded.loc[0, "transform"] == "call_center_degraded"
    assert degraded.loc[0, "target_sample_rate"] == 16000
    assert degraded.loc[0, "bandpass_hz"] == "300-3400"
```

- [ ] **Step 2: Run proxy data tests and verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_proxy_data.py -v
```

Expected output contains:

```text
FAILED tests/test_proxy_data.py - ModuleNotFoundError: No module named 'call_asr.proxy_data'
```

- [ ] **Step 3: Implement manifest validation and degradation metadata**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/proxy_data.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_PROXY_COLUMNS = ["file_name", "audio_path", "text", "source", "split"]


def validate_proxy_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    missing = [column for column in REQUIRED_PROXY_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Proxy manifest missing columns: {', '.join(missing)}")
    empty_text = df.loc[df["text"].astype(str) == "", "file_name"].tolist()
    if empty_text:
        raise ValueError(f"Proxy manifest has empty text: {', '.join(empty_text[:20])}")
    return df[REQUIRED_PROXY_COLUMNS].copy()


def degrade_manifest_for_call_center(df: pd.DataFrame) -> pd.DataFrame:
    degraded = df.copy()
    degraded["transform"] = "call_center_degraded"
    degraded["target_sample_rate"] = 16000
    degraded["bandpass_hz"] = "300-3400"
    degraded["mono"] = True
    degraded["gain_db"] = "-6,0,6"
    return degraded


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare proxy validation manifest metadata.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--clean-output", type=Path, required=True)
    parser.add_argument("--degraded-output", type=Path, required=True)
    args = parser.parse_args()
    clean = validate_proxy_manifest(args.input_manifest)
    degraded = degrade_manifest_for_call_center(clean)
    args.clean_output.parent.mkdir(parents=True, exist_ok=True)
    args.degraded_output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.clean_output, index=False)
    degraded.to_csv(args.degraded_output, index=False)
    print(f"Wrote clean manifest: {args.clean_output}")
    print(f"Wrote degraded manifest: {args.degraded_output}")
    print(f"Rows: {len(clean)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run proxy data tests and verify they pass**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_proxy_data.py -v
```

Expected output contains:

```text
tests/test_proxy_data.py::test_validate_proxy_manifest_accepts_required_columns PASSED
tests/test_proxy_data.py::test_degrade_manifest_for_call_center_adds_transform_columns PASSED
```

- [ ] **Step 5: Verify proxy manifest CLI with a deterministic two-row manifest**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
cat > /Users/temicide/Documents/5_domain_final/Call-ASR/data/proxy/public_thai_manifest.csv <<'CSV'
file_name,audio_path,text,source,split
cv_th_001.wav,/Users/temicide/Documents/5_domain_final/Call-ASR/data/proxy/audio/cv_th_001.wav,สวัสดีค่ะ,common_voice_th,validation
fleurs_th_001.wav,/Users/temicide/Documents/5_domain_final/Call-ASR/data/proxy/audio/fleurs_th_001.wav,โทรกลับพรุ่งนี้,fleurs_th,validation
CSV
python3 -m call_asr.proxy_data \
  --input-manifest /Users/temicide/Documents/5_domain_final/Call-ASR/data/proxy/public_thai_manifest.csv \
  --clean-output /Users/temicide/Documents/5_domain_final/Call-ASR/data/proxy/clean_manifest.csv \
  --degraded-output /Users/temicide/Documents/5_domain_final/Call-ASR/data/proxy/call_center_degraded_manifest.csv
```

Expected output contains:

```text
Wrote clean manifest: /Users/temicide/Documents/5_domain_final/Call-ASR/data/proxy/clean_manifest.csv
Wrote degraded manifest: /Users/temicide/Documents/5_domain_final/Call-ASR/data/proxy/call_center_degraded_manifest.csv
Rows: 2
```

- [ ] **Step 6: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
git add src/call_asr/proxy_data.py tests/test_proxy_data.py data/proxy/.gitkeep
git commit -m "feat: add proxy validation manifest hooks"
```

Expected output contains:

```text
[codex/
 feat: add proxy validation manifest hooks
```

### Task 10: Transcript Ensemble Utilities

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/ensemble.py`
- Test: `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_ensemble.py`

- [ ] **Step 1: Write failing ensemble tests**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_ensemble.py`:

```python
import pandas as pd

from call_asr.ensemble import confidence_select, prefix_route


def test_prefix_route_chooses_configured_model_by_file_prefix():
    typhoon = pd.DataFrame(
        {
            "file_name": ["AU_001_audio.wav", "RSP_001_audio.wav"],
            "normalized_text": ["typhoon au", "typhoon rsp"],
            "model_name": ["typhoon", "typhoon"],
            "avg_logprob": [-0.1, -0.1],
            "compression_ratio": [1.0, 1.0],
            "no_speech_prob": [0.01, 0.01],
        }
    )
    pathumma = pd.DataFrame(
        {
            "file_name": ["AU_001_audio.wav", "RSP_001_audio.wav"],
            "normalized_text": ["pathumma au", "pathumma rsp"],
            "model_name": ["pathumma", "pathumma"],
            "avg_logprob": [-0.2, -0.2],
            "compression_ratio": [1.0, 1.0],
            "no_speech_prob": [0.01, 0.01],
        }
    )

    routed = prefix_route(
        candidates={"typhoon": typhoon, "pathumma": pathumma},
        route_by_prefix={"AU": "pathumma"},
        default_model="typhoon",
    )

    assert routed[["file_name", "normalized_text", "selected_model"]].to_dict("records") == [
        {"file_name": "AU_001_audio.wav", "normalized_text": "pathumma au", "selected_model": "pathumma"},
        {"file_name": "RSP_001_audio.wav", "normalized_text": "typhoon rsp", "selected_model": "typhoon"},
    ]


def test_confidence_select_prefers_higher_logprob_then_lower_no_speech():
    model_a = pd.DataFrame(
        {
            "file_name": ["RSP_001_audio.wav"],
            "normalized_text": ["ก"],
            "model_name": ["a"],
            "avg_logprob": [-0.5],
            "compression_ratio": [1.1],
            "no_speech_prob": [0.01],
        }
    )
    model_b = pd.DataFrame(
        {
            "file_name": ["RSP_001_audio.wav"],
            "normalized_text": ["ข"],
            "model_name": ["b"],
            "avg_logprob": [-0.2],
            "compression_ratio": [1.2],
            "no_speech_prob": [0.03],
        }
    )

    selected = confidence_select({"a": model_a, "b": model_b})

    assert selected.loc[0, "normalized_text"] == "ข"
    assert selected.loc[0, "selected_model"] == "b"
```

- [ ] **Step 2: Run ensemble tests and verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_ensemble.py -v
```

Expected output contains:

```text
FAILED tests/test_ensemble.py - ModuleNotFoundError: No module named 'call_asr.ensemble'
```

- [ ] **Step 3: Implement ensemble utilities**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/src/call_asr/ensemble.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from call_asr.audio import audio_prefix


OUTPUT_COLUMNS = ["file_name", "normalized_text", "selected_model"]


def _candidate_row(frame: pd.DataFrame, file_name: str) -> pd.Series:
    matches = frame.loc[frame["file_name"] == file_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {file_name}, got {len(matches)}")
    return matches.iloc[0]


def prefix_route(
    candidates: dict[str, pd.DataFrame],
    route_by_prefix: dict[str, str],
    default_model: str,
) -> pd.DataFrame:
    if default_model not in candidates:
        raise ValueError(f"Default model not found in candidates: {default_model}")
    base = candidates[default_model]
    rows = []
    for file_name in base["file_name"].tolist():
        selected_model = route_by_prefix.get(audio_prefix(file_name), default_model)
        if selected_model not in candidates:
            raise ValueError(f"Route references missing model {selected_model} for {file_name}")
        candidate = _candidate_row(candidates[selected_model], file_name)
        rows.append(
            {
                "file_name": file_name,
                "normalized_text": candidate["normalized_text"],
                "selected_model": selected_model,
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def confidence_select(candidates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    first_model = next(iter(candidates))
    file_names = candidates[first_model]["file_name"].tolist()
    rows = []
    for file_name in file_names:
        scored_rows = []
        for model_name, frame in candidates.items():
            row = _candidate_row(frame, file_name).copy()
            row["selected_model"] = model_name
            scored_rows.append(row)
        selected = sorted(
            scored_rows,
            key=lambda row: (
                float(row.get("avg_logprob", 0.0)),
                -float(row.get("no_speech_prob", 0.0)),
                -float(row.get("compression_ratio", 0.0)),
            ),
            reverse=True,
        )[0]
        rows.append(
            {
                "file_name": file_name,
                "normalized_text": selected["normalized_text"],
                "selected_model": selected["selected_model"],
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine candidate ASR transcripts.")
    parser.add_argument("--method", choices=["confidence"], required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("MODEL_NAME", "CSV_PATH"), required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    candidates = {name: pd.read_csv(path, keep_default_na=False) for name, path in args.candidate}
    if args.method == "confidence":
        result = confidence_select(candidates)
    else:
        raise ValueError(f"Unsupported method: {args.method}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"Wrote ensemble: {args.output_csv}")
    print(f"Rows: {len(result)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run ensemble tests and verify they pass**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_ensemble.py -v
```

Expected output contains:

```text
tests/test_ensemble.py::test_prefix_route_chooses_configured_model_by_file_prefix PASSED
tests/test_ensemble.py::test_confidence_select_prefers_higher_logprob_then_lower_no_speech PASSED
```

- [ ] **Step 5: Run a confidence ensemble over two completed candidate CSVs**

Run after creating full candidate prediction files:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m call_asr.ensemble \
  --method confidence \
  --candidate typhoon /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/predictions_typhoon-ai__typhoon-whisper-large-v3_single_space.csv \
  --candidate pathumma /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/predictions_PogusTheWhisper__Pathumma-whisper-th-large-v3-natural-noise-finetuned_single_space.csv \
  --output-csv /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/ensemble_confidence_single_space.csv
```

Expected output contains:

```text
Wrote ensemble: /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/ensemble_confidence_single_space.csv
Rows: 6261
```

- [ ] **Step 6: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
git add src/call_asr/ensemble.py tests/test_ensemble.py
git commit -m "feat: add transcript ensemble selection"
```

Expected output contains:

```text
[codex/
 feat: add transcript ensemble selection
```

### Task 11: Colab Notebook Entrypoint

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/notebooks/colab_submission.py`
- Test: `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_colab_notebook_contract.py`

- [ ] **Step 1: Write failing static notebook contract tests**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_colab_notebook_contract.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SCRIPT = ROOT / "notebooks" / "colab_submission.py"


def test_notebook_script_writes_required_submission_path():
    source = NOTEBOOK_SCRIPT.read_text(encoding="utf-8")

    assert "/content/submission.csv" in source
    assert "/content/input/individual-test-thai-call-center-asr" in source
    assert "/content/working" in source
    assert "write_submission_csv" in source
    assert "validate_submission_frame" in source


def test_notebook_script_downloads_and_extracts_before_reading():
    source = NOTEBOOK_SCRIPT.read_text(encoding="utf-8")

    assert "configure_kaggle_credentials" in source
    assert "download_and_extract_competition_data" in source
    assert "kaggle competitions download" in source or "competition_download_files" in source
    assert ".extractall" in source
    assert source.index("download_and_extract_competition_data(") < source.index("resolve_competition_paths(")


def test_notebook_script_does_not_print_credentials():
    source = NOTEBOOK_SCRIPT.read_text(encoding="utf-8").lower()
    assert "print(kaggle_username" not in source
    assert "print(kaggle_key" not in source
    assert "print(credentials" not in source


def test_notebook_script_does_not_submit_to_kaggle():
    source = NOTEBOOK_SCRIPT.read_text(encoding="utf-8").lower()
    forbidden = [
        "kaggle competitions submit",
        ".competitions.submit",
        "api.competition_submit",
        "competition_submit(",
    ]

    assert [needle for needle in forbidden if needle in source] == []
```

- [ ] **Step 2: Run notebook contract tests and verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_colab_notebook_contract.py -v
```

Expected output contains:

```text
FAILED tests/test_colab_notebook_contract.py::test_notebook_script_writes_required_submission_path - FileNotFoundError
```

- [ ] **Step 3: Implement the Colab notebook script**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/notebooks/colab_submission.py`:

```python
# %% [markdown]
# # Thai Call Center ASR Submission
# This notebook downloads Kaggle competition data, generates /content/submission.csv,
# and stops before any upload or submission step.

# %%
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import zipfile

COMPETITION_SLUG = "individual-test-thai-call-center-asr"
CONTENT_ROOT = Path("/content")
INPUT_DIR = CONTENT_ROOT / "input" / COMPETITION_SLUG
WORKING_DIR = CONTENT_ROOT / "working"
SUBMISSION_PATH = CONTENT_ROOT / "submission.csv"
PROJECT_ROOT = CONTENT_ROOT / "Call-ASR"
LOCAL_PROJECT_ROOT = Path("/Users/temicide/Documents/5_domain_final/Call-ASR")
if LOCAL_PROJECT_ROOT.exists():
    PROJECT_ROOT = LOCAL_PROJECT_ROOT
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def configure_kaggle_credentials() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json_path = kaggle_dir / "kaggle.json"

    uploaded_json = Path("/content/kaggle.json")
    if uploaded_json.exists():
        shutil.copyfile(uploaded_json, kaggle_json_path)
    else:
        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")
        if not username or not key:
            try:
                from google.colab import userdata

                username = username or userdata.get("KAGGLE_USERNAME")
                key = key or userdata.get("KAGGLE_KEY")
            except Exception:
                pass
        if not username or not key:
            raise RuntimeError(
                "Provide Kaggle credentials via uploaded /content/kaggle.json "
                "or Colab secrets/environment variables KAGGLE_USERNAME and KAGGLE_KEY."
            )
        kaggle_json_path.write_text(
            json.dumps({"username": username, "key": key}),
            encoding="utf-8",
        )
    kaggle_json_path.chmod(0o600)


def download_and_extract_competition_data() -> Path:
    configure_kaggle_credentials()
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_submission = INPUT_DIR / "sample_submission.csv"
    audio_dir = INPUT_DIR / "audio_final" / "audio"
    if sample_submission.exists() and audio_dir.is_dir():
        return INPUT_DIR

    archive_path = WORKING_DIR / f"{COMPETITION_SLUG}.zip"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION_SLUG,
            "-p",
            str(WORKING_DIR),
            "--force",
        ],
        check=True,
    )
    if not archive_path.exists():
        raise FileNotFoundError(f"Downloaded archive not found: {archive_path}")
    with zipfile.ZipFile(archive_path) as zip_file:
        zip_file.extractall(INPUT_DIR)
    return INPUT_DIR

# %%
from call_asr.infer import WhisperPipelineBackend, run_inference
from call_asr.paths import resolve_competition_paths
from call_asr.submission import (
    load_sample_submission,
    validate_audio_coverage,
    validate_submission_frame,
    write_submission_csv,
)

# %%
MODEL_NAME = "typhoon-ai/typhoon-whisper-large-v3"
NORMALIZATION_POLICY = "single_space"
CHUNK_LENGTH_SECONDS = 30
BATCH_SIZE = 4
CONDITION_ON_PREVIOUS_TEXT = False

# %%
download_and_extract_competition_data()
paths = resolve_competition_paths(colab_input_root=CONTENT_ROOT / "input", project_root=PROJECT_ROOT)
sample_df = load_sample_submission(paths.sample_submission)
validate_audio_coverage(sample_df, paths.audio_dir)
print(f"Rows in sample submission: {len(sample_df)}")
print(f"Audio directory: {paths.audio_dir}")
print(f"Output submission path: {SUBMISSION_PATH}")

# %%
backend = WhisperPipelineBackend(
    model_name=MODEL_NAME,
    chunk_length_s=CHUNK_LENGTH_SECONDS,
    batch_size=BATCH_SIZE,
    condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
    device="cuda",
)

# %%
safe_model_name = MODEL_NAME.replace("/", "__")
predictions_path = paths.working_dir / f"predictions_{safe_model_name}_{NORMALIZATION_POLICY}.csv"
log_path = paths.working_dir / f"run_{safe_model_name}_{NORMALIZATION_POLICY}.jsonl"
predictions_df = run_inference(
    sample_df=sample_df,
    audio_dir=paths.audio_dir,
    output_csv=predictions_path,
    log_jsonl=log_path,
    backend=backend,
    normalization_policy=NORMALIZATION_POLICY,
    resume=True,
)
print(f"Wrote predictions: {predictions_path}")
print(f"Wrote logs: {log_path}")
print(f"Prediction rows: {len(predictions_df)}")
print(f"Inference errors: {(predictions_df['error'] != '').sum()}")

# %%
submission_df = write_submission_csv(
    sample_df=sample_df,
    predictions_df=predictions_df,
    output_path=SUBMISSION_PATH,
    allow_empty_files=set(),
)
validate_submission_frame(submission_df, allow_empty_files=set())
print(f"Wrote validated submission: {SUBMISSION_PATH}")
print(submission_df.head().to_string(index=False))
```

- [ ] **Step 4: Run notebook contract tests and verify they pass**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_colab_notebook_contract.py -v
```

Expected output contains:

```text
tests/test_colab_notebook_contract.py::test_notebook_script_writes_required_submission_path PASSED
tests/test_colab_notebook_contract.py::test_notebook_script_downloads_and_extracts_before_reading PASSED
tests/test_colab_notebook_contract.py::test_notebook_script_does_not_print_credentials PASSED
tests/test_colab_notebook_contract.py::test_notebook_script_does_not_submit_to_kaggle PASSED
```

- [ ] **Step 5: Run a static all-tests check before using Kaggle**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest -v
```

Expected output contains:

```text
tests/test_audio.py
tests/test_audit_audio.py
tests/test_ensemble.py
tests/test_infer_resume.py
tests/test_colab_notebook_contract.py
tests/test_normalize_text.py
tests/test_paths.py
tests/test_project_scaffold.py
tests/test_proxy_data.py
tests/test_score_proxy.py
tests/test_submission.py
```

Expected final line contains:

```text
passed
```

- [ ] **Step 6: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
git add notebooks/colab_submission.py tests/test_colab_notebook_contract.py
git commit -m "feat: add colab submission notebook entrypoint"
```

Expected output contains:

```text
[codex/
 feat: add colab submission notebook entrypoint
```

### Task 12: Final Submission Generation and Documentation

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Call-ASR/README.md`
- Test: `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_readme_contract.py`

- [ ] **Step 1: Write failing README contract tests**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/tests/test_readme_contract.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_manual_submission_boundary_and_commands():
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "/kaggle/working/submission.csv" in text
    assert "python3 -m call_asr.audit_audio" in text
    assert "python3 -m call_asr.infer" in text
    assert "python3 -m pytest -v" in text
    assert "Do not run kaggle competitions submit" in text
```

- [ ] **Step 2: Run README test and verify it fails**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_readme_contract.py -v
```

Expected output contains:

```text
FAILED tests/test_readme_contract.py::test_readme_documents_manual_submission_boundary_and_commands - FileNotFoundError
```

- [ ] **Step 3: Add README with exact commands and artifact paths**

Create `/Users/temicide/Documents/5_domain_final/Call-ASR/README.md`:

```markdown
# Thai Call Center ASR

This project generates a Kaggle-ready Thai call-center ASR submission CSV for `individual-test-thai-call-center-asr`.

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

## Kaggle Notebook

Upload or paste `/Users/temicide/Documents/5_domain_final/Call-ASR/notebooks/kaggle_submission.py` into a Kaggle Notebook with GPU enabled. The notebook reads competition files from `/kaggle/input/individual-test-thai-call-center-asr`, validates the sample submission schema and audio coverage, runs resumable inference, and writes:

```text
/kaggle/working/submission.csv
```

Do not run kaggle competitions submit. Manual download or normal Kaggle notebook output use is the allowed boundary for the generated CSV.
```

- [ ] **Step 4: Run README test and verify it passes**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest tests/test_readme_contract.py -v
```

Expected output contains:

```text
tests/test_readme_contract.py::test_readme_documents_manual_submission_boundary_and_commands PASSED
```

- [ ] **Step 5: Generate the final Kaggle submission locally from Typhoon single-space predictions**

Run after `/Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/predictions_typhoon-ai__typhoon-whisper-large-v3_single_space.csv` has `6261` rows and zero inference errors:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 - <<'PY'
from pathlib import Path
import pandas as pd
from call_asr.paths import resolve_competition_paths
from call_asr.submission import load_sample_submission, write_submission_csv, validate_submission_frame

paths = resolve_competition_paths(project_root=Path('/Users/temicide/Documents/5_domain_final/Call-ASR'))
sample_df = load_sample_submission(paths.sample_submission)
predictions_df = pd.read_csv(
    '/Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/predictions_typhoon-ai__typhoon-whisper-large-v3_single_space.csv',
    keep_default_na=False,
)
submission_df = write_submission_csv(
    sample_df=sample_df,
    predictions_df=predictions_df,
    output_path=Path('/Users/temicide/Documents/5_domain_final/Call-ASR/data/submissions/sub_typhoon_single_space.csv'),
    allow_empty_files=set(),
)
validate_submission_frame(submission_df, allow_empty_files=set())
print(f'rows={len(submission_df)}')
print(f'columns={list(submission_df.columns)}')
print(submission_df.head().to_string(index=False))
PY
```

Expected output contains:

```text
rows=6261
columns=['file_name', 'text']
```

- [ ] **Step 6: Run full verification**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m pytest -v
```

Expected final line contains:

```text
passed
```

- [ ] **Step 7: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
git add README.md tests/test_readme_contract.py
git commit -m "docs: document call asr execution workflow"
```

Expected output contains:

```text
[codex/
 docs: document call asr execution workflow
```

## Experiment Sequence

Use this sequence after the harness passes tests. Each candidate generates a full `6261`-row prediction CSV, then a validated submission CSV. Upload only manually through Kaggle UI or the standard Kaggle notebook output panel.

1. Run `typhoon-ai/typhoon-whisper-large-v3` with `single_space`.

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m call_asr.infer \
  --model-name typhoon-ai/typhoon-whisper-large-v3 \
  --normalization-policy single_space \
  --chunk-length-s 30 \
  --batch-size 4
```

Expected output contains:

```text
Rows: 6261
Errors: 0
```

2. Reuse the same raw transcripts to generate `no_spaces` and `thai_chars_only_light` variants by rerunning inference with the same model and each normalization policy.

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m call_asr.infer \
  --model-name typhoon-ai/typhoon-whisper-large-v3 \
  --normalization-policy no_spaces \
  --chunk-length-s 30 \
  --batch-size 4
python3 -m call_asr.infer \
  --model-name typhoon-ai/typhoon-whisper-large-v3 \
  --normalization-policy thai_chars_only_light \
  --chunk-length-s 30 \
  --batch-size 4
```

Expected output for each run contains:

```text
Rows: 6261
Errors: 0
```

3. Run diversity models with `single_space` normalization to match the baseline artifact names used by the confidence ensemble command.

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m call_asr.infer \
  --model-name biodatlab/whisper-th-large-v3-combined \
  --normalization-policy single_space \
  --chunk-length-s 30 \
  --batch-size 4
python3 -m call_asr.infer \
  --model-name PogusTheWhisper/Pathumma-whisper-th-large-v3-natural-noise-finetuned \
  --normalization-policy single_space \
  --chunk-length-s 30 \
  --batch-size 4
python3 -m call_asr.infer \
  --model-name openai/whisper-large-v3 \
  --normalization-policy single_space \
  --chunk-length-s 30 \
  --batch-size 4
```

Expected output for each run contains:

```text
Rows: 6261
Errors: 0
```

4. Build a confidence ensemble over completed candidate files.

```bash
cd /Users/temicide/Documents/5_domain_final/Call-ASR
python3 -m call_asr.ensemble \
  --method confidence \
  --candidate typhoon /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/predictions_typhoon-ai__typhoon-whisper-large-v3_single_space.csv \
  --candidate thonburian /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/predictions_biodatlab__whisper-th-large-v3-combined_single_space.csv \
  --candidate pathumma /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/predictions_PogusTheWhisper__Pathumma-whisper-th-large-v3-natural-noise-finetuned_single_space.csv \
  --output-csv /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/ensemble_confidence_single_space.csv
```

Expected output contains:

```text
Wrote ensemble: /Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/ensemble_confidence_single_space.csv
Rows: 6261
```

5. Record every manual leaderboard result in `/Users/temicide/Documents/5_domain_final/Call-ASR/data/runs/leaderboard_log.csv` with these columns:

```csv
submission_id,timestamp,models,decode_params,normalization,score,diff_from_previous,decision
manual_001,2026-06-06T00:00:00Z,typhoon-ai/typhoon-whisper-large-v3,chunk=30 batch=4 condition_on_previous_text=false,single_space,4.53538,0.00000,baseline sanity check against old visible metadata
```

## Self-Review Notes

**Spec coverage:** This plan covers the Kaggle Notebook requirement in Task 11, the exact sample submission schema and row-order preservation in Task 3, local audio counts and prefix handling in Tasks 4 and 5, resumable model inference and logging in Task 8, Thai-safe normalization in Task 6, proxy validation metrics and manifests in Tasks 7 and 9, ensemble methods in Task 10, public leaderboard discipline in the Experiment Sequence, and documentation/manual submission boundaries in Task 12.

**Placeholder scan:** The plan avoids unresolved tokens and vague implementation directives. Every code-editing step includes concrete file content, exact commands, and expected output fragments.

**Type consistency:** The shared names are consistent across tasks: `CompetitionPaths`, `load_sample_submission`, `validate_audio_coverage`, `validate_submission_frame`, `write_submission_csv`, `AudioMetadata`, `AsrResult`, `FakeAsrBackend`, `WhisperPipelineBackend`, `run_inference`, `normalize_text`, `score_predictions`, `prefix_route`, and `confidence_select`.

**Execution handoff:** Plan complete and saved to `/Users/temicide/Documents/5_domain_final/Call-ASR/plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.
