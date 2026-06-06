# Sleep-Stage Colab Solution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Google Colab-ready solution that downloads and extracts Kaggle competition data inside Colab, trains grouped-validation sleep-stage models, generates a validated `/content/submission.csv`, and avoids automated competition submission.

**Architecture:** Use a focused Python package under `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage` for data bootstrap, data loading, feature engineering, grouped validation, context features, smoothing, and submission generation. Keep `/Users/temicide/Documents/5_domain_final/Sleep-Stage/sleep_stage_solution.ipynb` as the Colab-ready executable wrapper that calls the same package logic, with local path fallbacks for development. Model selection is driven by `GroupKFold` over full train recordings, then the final model predicts test segment IDs from `sample_submission.csv`.

**Tech Stack:** Python 3.10+, NumPy, pandas, scikit-learn, SciPy, joblib, pytest, optional LightGBM when already available or installable in the Colab runtime.

---

## File Structure

- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/pyproject.toml`: local package metadata, pytest configuration, and runtime dependencies.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/__init__.py`: package marker and version string.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/config.py`: constants for labels, paths, sampling rate, epoch length, Colab path discovery, and local fallback discovery.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/kaggle_download.py`: Colab-safe credential handling, Kaggle CLI/API download, archive extraction, and verification that files exist before reads.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/data.py`: train recording loading, test segment loading, epoch splitting, sample-submission mapping, and CSV submission validation.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/features.py`: deterministic per-epoch feature extraction for time statistics, motion, HR/IBI, BVP spectrum, and cross-signal features.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/context.py`: previous/next epoch context, rolling statistics, epoch deltas, and relative-position features.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/models.py`: model factories, grouped CV runner, fold reports, final fit, and probability blending.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/smoothing.py`: mode filters and Viterbi decoding from train transition counts.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/pipeline.py`: command-style orchestration for cache building, validation experiments, and submission generation.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_data.py`: synthetic tests for path discovery, data bootstrap verification, epoch splitting, and submission validation.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_features_context.py`: synthetic tests for feature names, finite values, and context feature shape.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_models_smoothing.py`: synthetic tests for grouped validation plumbing and smoothing behavior.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/sleep_stage_solution.ipynb`: Colab-ready solution that configures Kaggle credentials, downloads and extracts data into `/content/input`, trains the best validated pipeline, validates predictions, and writes `/content/submission.csv`.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/submission_log.md`: manual record of local CV, public LB, and change description for each manually uploaded candidate.

## Task 1: Package Skeleton, Constants, And Path Discovery

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/pyproject.toml`
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/__init__.py`
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/config.py`
- Test: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_data.py`

- [ ] **Step 1: Write failing tests for constants and path discovery**

```python
# /Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_data.py
from pathlib import Path

import pandas as pd

from sleep_stage.config import (
    COLAB_INPUT_ROOT,
    COLAB_OUTPUT_PATH,
    COMPETITION_SLUG,
    EPOCH_ROWS,
    LABELS,
    SIGNAL_COLUMNS,
    find_competition_root,
    build_paths,
)
from sleep_stage.data import validate_submission


def test_constants_match_competition_spec():
    assert EPOCH_ROWS == 480
    assert SIGNAL_COLUMNS == ["BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA", "HR", "IBI"]
    assert LABELS == ["W", "N1", "N2", "N3", "R"]
    assert COMPETITION_SLUG == "super-ai-engineer-ss-6-individual-sleep-stage-classification"
    assert COLAB_INPUT_ROOT == Path("/content/input")
    assert COLAB_OUTPUT_PATH == Path("/content/submission.csv")


def test_find_competition_root_prefers_colab_then_local(tmp_path: Path):
    colab_root = tmp_path / "content" / "input" / COMPETITION_SLUG
    local_root = tmp_path / "local" / "data" / COMPETITION_SLUG
    colab_root.mkdir(parents=True)
    local_root.mkdir(parents=True)

    found = find_competition_root(
        colab_input_root=tmp_path / "content" / "input",
        local_data_root=tmp_path / "local" / "data",
    )

    assert found == colab_root


def test_build_paths_uses_colab_output_when_content_exists(tmp_path: Path):
    competition_root = tmp_path / "input" / COMPETITION_SLUG
    competition_root.mkdir(parents=True)
    paths = build_paths(
        project_root=tmp_path / "project",
        competition_root=competition_root,
        output_path=tmp_path / "submission.csv",
        working_dir=tmp_path / "working",
    )

    assert paths.competition_root == competition_root
    assert paths.output_path == tmp_path / "submission.csv"
    assert paths.working_dir == tmp_path / "working"


def test_validate_submission_accepts_complete_known_labels(tmp_path: Path):
    sample = pd.DataFrame({"id": ["test001_00000", "test001_00001"], "labels": ["N2", "N2"]})
    submission = pd.DataFrame({"id": ["test001_00000", "test001_00001"], "labels": ["W", "R"]})
    sample_path = tmp_path / "sample_submission.csv"
    output_path = tmp_path / "submission.csv"
    sample.to_csv(sample_path, index=False)
    submission.to_csv(output_path, index=False)

    result = validate_submission(output_path, sample_path)

    assert result.rows == 2
    assert result.valid_labels == ["W", "R"]
    assert result.id_match is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_data.py -v
```

Expected output:

```text
E   ModuleNotFoundError: No module named 'sleep_stage'
```

- [ ] **Step 3: Add package metadata and constants**

```toml
# /Users/temicide/Documents/5_domain_final/Sleep-Stage/pyproject.toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sleep-stage"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "joblib>=1.3",
  "numpy>=1.23",
  "pandas>=1.5",
  "scikit-learn>=1.2",
  "scipy>=1.9",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-q"
```

```python
# /Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/__init__.py
__version__ = "0.1.0"
```

```python
# /Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

COMPETITION_SLUG = "super-ai-engineer-ss-6-individual-sleep-stage-classification"
SAMPLING_HZ = 16
EPOCH_SECONDS = 30
EPOCH_ROWS = SAMPLING_HZ * EPOCH_SECONDS
SIGNAL_COLUMNS = ["BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA", "HR", "IBI"]
LABEL_COLUMN = "Sleep_Stage"
LABELS = ["W", "N1", "N2", "N3", "R"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}

COLAB_ROOT = Path("/content")
COLAB_INPUT_ROOT = COLAB_ROOT / "input"
COLAB_WORKING_DIR = COLAB_ROOT / "working"
COLAB_OUTPUT_PATH = COLAB_ROOT / "submission.csv"


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    competition_root: Path
    train_dir: Path
    test_dir: Path
    sample_submission: Path
    cache_dir: Path
    working_dir: Path
    output_path: Path


def find_competition_root(
    colab_input_root: Path = COLAB_INPUT_ROOT,
    local_data_root: Path | None = None,
) -> Path:
    local_root = local_data_root or Path(__file__).resolve().parents[2] / "data"
    candidates = [
        colab_input_root / COMPETITION_SLUG,
        local_root / COMPETITION_SLUG,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Competition data not found. Download and extract it in Colab first, or provide local data. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def build_paths(
    project_root: Path | None = None,
    competition_root: Path | None = None,
    output_path: Path | None = None,
    working_dir: Path | None = None,
) -> ProjectPaths:
    root = project_root or Path(__file__).resolve().parents[2]
    data_root = competition_root or find_competition_root(local_data_root=root / "data")
    work = working_dir or (COLAB_WORKING_DIR if COLAB_ROOT.exists() else root / "working")
    out = output_path or (COLAB_OUTPUT_PATH if COLAB_ROOT.exists() else root / "working" / "submission.csv")
    return ProjectPaths(
        project_root=root,
        competition_root=data_root,
        train_dir=data_root / "train" / "train",
        test_dir=data_root / "test_segment" / "test_segment",
        sample_submission=data_root / "sample_submission.csv",
        cache_dir=work / "cache",
        working_dir=work,
        output_path=out,
    )
```

- [ ] **Step 4: Add the submission validator used by the tests**

```python
# /Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sleep_stage.config import LABELS


@dataclass(frozen=True)
class SubmissionValidation:
    rows: int
    valid_labels: list[str]
    id_match: bool


def validate_submission(submission_path: Path, sample_submission_path: Path) -> SubmissionValidation:
    submission = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_submission_path)
    expected_columns = ["id", "labels"]
    if list(submission.columns) != expected_columns:
        raise ValueError(f"submission columns must be {expected_columns}, found {list(submission.columns)}")
    if len(submission) != len(sample):
        raise ValueError(f"submission row count {len(submission)} does not match sample row count {len(sample)}")
    id_match = submission["id"].astype(str).tolist() == sample["id"].astype(str).tolist()
    if not id_match:
        raise ValueError("submission ids must exactly match sample_submission.csv order")
    invalid = sorted(set(submission["labels"].astype(str)) - set(LABELS))
    if invalid:
        raise ValueError(f"submission contains labels outside {LABELS}: {invalid}")
    if submission["labels"].isna().any():
        raise ValueError("submission labels contain missing values")
    return SubmissionValidation(
        rows=len(submission),
        valid_labels=submission["labels"].astype(str).tolist(),
        id_match=True,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_data.py -v
```

Expected output:

```text
tests/test_data.py::test_constants_match_competition_spec PASSED
tests/test_data.py::test_find_competition_root_prefers_colab_then_local PASSED
tests/test_data.py::test_build_paths_uses_colab_output_when_content_exists PASSED
tests/test_data.py::test_validate_submission_accepts_complete_known_labels PASSED
```

## Task 2: Colab Kaggle Download And Extraction

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/kaggle_download.py`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_data.py`

- [ ] **Step 1: Write failing tests for bootstrap verification**

```python
# Append to /Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_data.py
from sleep_stage.kaggle_download import verify_extracted_competition_data


def test_verify_extracted_competition_data_requires_expected_files(tmp_path: Path):
    root = tmp_path / COMPETITION_SLUG
    (root / "train" / "train").mkdir(parents=True)
    (root / "test_segment" / "test_segment" / "test001").mkdir(parents=True)
    (root / "sample_submission.csv").write_text("id,labels\n")

    verified = verify_extracted_competition_data(root)

    assert verified == root
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_data.py::test_verify_extracted_competition_data_requires_expected_files -v
```

Expected output:

```text
E   ModuleNotFoundError: No module named 'sleep_stage.kaggle_download'
```

- [ ] **Step 3: Implement Colab-safe Kaggle credential and download helpers**

```python
# /Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/kaggle_download.py
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path

from sleep_stage.config import COLAB_INPUT_ROOT, COMPETITION_SLUG


def configure_kaggle_credentials(kaggle_json_path: Path | None = None) -> Path | None:
    """Configure credentials without printing secret values."""
    target = Path.home() / ".kaggle" / "kaggle.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    if kaggle_json_path is not None and kaggle_json_path.exists():
        shutil.copy2(kaggle_json_path, target)
    elif os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        target.write_text(
            json.dumps(
                {
                    "username": os.environ["KAGGLE_USERNAME"],
                    "key": os.environ["KAGGLE_KEY"],
                }
            )
        )
    elif target.exists():
        pass
    else:
        return None

    target.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return target


def verify_extracted_competition_data(competition_root: Path) -> Path:
    required = [
        competition_root / "train" / "train",
        competition_root / "test_segment" / "test_segment",
        competition_root / "sample_submission.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Extracted competition data is incomplete: " + ", ".join(str(path) for path in missing))
    return competition_root


def download_and_extract_competition(
    input_root: Path = COLAB_INPUT_ROOT,
    competition_slug: str = COMPETITION_SLUG,
    kaggle_json_path: Path | None = None,
) -> Path:
    credential_path = configure_kaggle_credentials(kaggle_json_path)
    if credential_path is None:
        raise RuntimeError(
            "Kaggle credentials not found. Upload kaggle.json or set KAGGLE_USERNAME and KAGGLE_KEY in Colab secrets/environment."
        )

    input_root.mkdir(parents=True, exist_ok=True)
    archive_path = input_root / f"{competition_slug}.zip"
    extracted_root = input_root / competition_slug
    if not archive_path.exists():
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", competition_slug, "-p", str(input_root)],
            check=True,
        )

    extracted_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted_root)

    return verify_extracted_competition_data(extracted_root)
```

- [ ] **Step 4: Run tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_data.py -v
```

Expected output:

```text
tests/test_data.py::test_verify_extracted_competition_data_requires_expected_files PASSED
```

## Task 3: Data Loading And Epoch Splitting

**Files:**
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/data.py`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_data.py`

- [ ] **Step 1: Add tests for epoch conversion and test ID mapping**

Use synthetic frames to verify:

- `epoch_dataframe_from_recording` returns one row per 480 samples.
- Labels are constant per epoch.
- `list_test_segment_paths` finds `test*/test*.csv` files.
- `read_test_segments` follows `sample_submission.csv` order.
- Test segments must contain exactly 480 rows and all signal columns.

- [ ] **Step 2: Implement loading and epoch helpers**

Implement:

- `_validate_signal_columns(frame, require_label)`.
- `epoch_dataframe_from_recording(frame, recording_id)`.
- `load_train_recording(path)`.
- `iter_train_recordings(train_dir)`.
- `list_test_segment_paths(test_root)`.
- `read_test_segments(sample_submission_path, segment_paths)`.

- [ ] **Step 3: Run tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_data.py -v
```

Expected output includes:

```text
test_epoch_dataframe_from_recording_returns_one_row_per_480_samples PASSED
test_list_and_read_test_segments_follow_sample_submission_order PASSED
```

## Task 4: Per-Epoch Feature Extraction

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/features.py`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_features_context.py`

- [ ] **Step 1: Add feature tests**

Verify:

- `extract_epoch_features` returns deterministic sorted feature names.
- All feature values are finite.
- Required features include `BVP_fft_peak_hz`, `ACC_mag_mean`, and `HR_IBI_consistency`.
- `extract_feature_table` preserves metadata and sorts feature columns.

- [ ] **Step 2: Implement deterministic feature extraction**

Feature families:

- Per-channel mean, std, min, max, median, quantiles, IQR, skew, kurtosis, range, slope, mean absolute difference, max absolute difference, and zero-crossing proxy.
- ACC magnitude and jerk.
- BVP FFT band-power ratios and peak frequency.
- HR and IBI RMSSD-like features.
- Cross-signal features such as HR/IBI consistency, ACC magnitude x HR, and EDA x TEMP.

- [ ] **Step 3: Run tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_features_context.py -v
```

Expected output includes:

```text
test_extract_epoch_features_has_deterministic_finite_values PASSED
test_extract_feature_table_returns_metadata_and_sorted_feature_columns PASSED
```

## Task 5: Context And Rolling Features

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/context.py`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_features_context.py`

- [ ] **Step 1: Add context tests**

Verify:

- Row count is preserved.
- Context is computed within each recording or test subject only.
- Previous and next lag features fill boundaries with the current value.
- Rolling mean/std features are present.
- `relative_position` is 0 for first epoch and 1 for last epoch when a sequence has more than one epoch.

- [ ] **Step 2: Implement temporal context features**

Implement:

- `METADATA_COLUMNS`.
- `feature_columns(table)`.
- `add_context_features(table, lags=(1, 2, 3), rolling_windows=(3, 5, 9, 15))`.

- [ ] **Step 3: Run tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_features_context.py -v
```

Expected output includes:

```text
test_add_context_features_preserves_rows_and_adds_temporal_columns PASSED
```

## Task 6: Grouped Cross-Validation Models

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/models.py`
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_models_smoothing.py`

- [ ] **Step 1: Add grouped-CV tests**

Verify:

- `build_model("extra_trees")` returns a `fit`/`predict_proba` estimator.
- `run_grouped_cv` uses `GroupKFold`, returns 5 fold scores, per-class F1 for all labels, confusion matrix, feature columns, and out-of-fold probabilities with shape `(n_rows, 5)`.

- [ ] **Step 2: Implement model factories and grouped validation**

Models:

- `HistGradientBoostingClassifier` as `hgb`.
- `ExtraTreesClassifier` as `extra_trees`.
- `RandomForestClassifier` as `random_forest`.

Validation output must include weighted F1 mean/std, per-class F1 for W/N1/N2/N3/R, and confusion matrix.

- [ ] **Step 3: Run tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_models_smoothing.py -v
```

Expected output includes:

```text
test_build_model_returns_predict_proba_estimator PASSED
test_run_grouped_cv_returns_fold_metrics_and_oof_probabilities PASSED
```

## Task 7: Temporal Smoothing And Viterbi Decoding

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/smoothing.py`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_models_smoothing.py`

- [ ] **Step 1: Add smoothing tests**

Verify:

- `mode_filter_labels` removes a one-epoch spike.
- `build_transition_log_probs` returns a 5x5 transition matrix.
- `viterbi_decode` accepts probability arrays with shape `(n_epochs, 5)` and returns legal label strings.

- [ ] **Step 2: Implement smoothing utilities**

Implement:

- `mode_filter_labels(labels, window=3)`.
- `build_transition_log_probs(train_sequences, smoothing=1.0)`.
- `viterbi_decode(probabilities, transition_log_probs)`.

- [ ] **Step 3: Run tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_models_smoothing.py -v
```

Expected output includes:

```text
test_mode_filter_labels_removes_single_epoch_spike PASSED
test_viterbi_decode_uses_transition_counts_and_probability_shape PASSED
```

## Task 8: Feature Cache, Experiments, And Submission Pipeline

**Files:**
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/data.py`
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/pipeline.py`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_data.py`

- [ ] **Step 1: Add pipeline tests**

Verify:

- `write_submission_from_labels` preserves sample submission order.
- It writes exactly `id,labels`.
- It validates row count, ID order, legal labels, and missing values.

- [ ] **Step 2: Implement pipeline orchestration**

Implement:

- `build_train_feature_table(paths)`.
- `build_test_feature_table(paths)`.
- `load_or_build_features(paths, force=False)` using `paths.cache_dir`.
- `train_sequences_from_table(table)`.
- `predict_labels_by_segment(train_table, test_table, model_name, use_context, use_viterbi)`.
- `write_submission_from_labels(labels_by_id, sample_submission_path, output_path)`.
- `run_experiments(paths=None, force_features=False)`.
- `generate_submission(paths=None)` writing to `paths.output_path`.

The Colab output path must resolve to `/content/submission.csv`. The local fallback output path must resolve to `/Users/temicide/Documents/5_domain_final/Sleep-Stage/working/submission.csv`.

- [ ] **Step 3: Run full unit tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest -v
```

Expected output:

```text
13 passed
```

## Task 9: Local Full-Data Experiment Run

**Files:**
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/working/cache/experiment_results.csv`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/working/cache/*_cv_report.joblib`

- [ ] **Step 1: Build local feature caches**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python - <<'PY'
from sleep_stage.config import build_paths
from sleep_stage.pipeline import load_or_build_features

paths = build_paths()
train, test = load_or_build_features(paths, force=True)
print(train.shape)
print(test.shape)
print(train["label"].value_counts().to_dict())
print(paths.competition_root)
print(paths.cache_dir)
PY
```

Expected output:

```text
(66745, ...)
(7832, ...)
{'N2': 33786, 'W': 15828, 'N1': 7753, 'R': 7033, 'N3': 2345}
/Users/temicide/Documents/5_domain_final/Sleep-Stage/data/super-ai-engineer-ss-6-individual-sleep-stage-classification
/Users/temicide/Documents/5_domain_final/Sleep-Stage/working/cache
```

- [ ] **Step 2: Run grouped CV experiments**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python - <<'PY'
from sleep_stage.pipeline import run_experiments

results = run_experiments(force_features=False)
print(results[["experiment", "model_name", "n_features", "weighted_f1_mean", "weighted_f1_std"]].to_string(index=False))
PY
```

Expected output includes:

```text
        experiment   model_name  n_features  weighted_f1_mean  weighted_f1_std
       context_hgb          hgb
context_extra_trees  extra_trees
        static_hgb          hgb
 static_extra_trees  extra_trees
```

If `context_hgb` scores below `0.50`, inspect the saved report for per-class collapse before producing a leaderboard-candidate file.

- [ ] **Step 3: Print per-class F1 and confusion matrix for the best experiment**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python - <<'PY'
import joblib
from sleep_stage.config import LABELS

report = joblib.load("working/cache/context_hgb_cv_report.joblib")
print(report.per_class_f1.to_string(index=False))
print("fold_scores=", [round(score, 5) for score in report.fold_scores])
print("mean=", round(report.weighted_f1_mean, 5), "std=", round(report.weighted_f1_std, 5))
print("confusion labels=", LABELS)
print(report.confusion)
PY
```

Expected output:

```text
 fold        W       N1       N2       N3        R
    1
    2
    3
    4
    5
fold_scores=
mean=
confusion labels= ['W', 'N1', 'N2', 'N3', 'R']
```

## Task 10: Colab Notebook Wrapper

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/sleep_stage_solution.ipynb`

- [ ] **Step 1: Create the notebook with these cells**

Cell 1: install/import prerequisites and set paths.

```python
from pathlib import Path
import os
import sys

try:
    import kaggle  # noqa: F401
except Exception:
    %pip -q install kaggle

PROJECT_ROOT = Path("/content/Sleep-Stage")
LOCAL_ROOT = Path("/Users/temicide/Documents/5_domain_final/Sleep-Stage")
if LOCAL_ROOT.exists():
    PROJECT_ROOT = LOCAL_ROOT

src_path = PROJECT_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

CONTENT_ROOT = Path("/content")
INPUT_ROOT = CONTENT_ROOT / "input"
WORKING_DIR = CONTENT_ROOT / "working"
SUBMISSION_PATH = CONTENT_ROOT / "submission.csv"
INPUT_ROOT.mkdir(parents=True, exist_ok=True)
WORKING_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using project root: {PROJECT_ROOT}")
```

Cell 2: configure Kaggle credentials without printing secrets.

```python
from pathlib import Path
import os

uploaded_kaggle_json = None
if not (Path.home() / ".kaggle" / "kaggle.json").exists():
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        from google.colab import files
        uploaded = files.upload()
        if "kaggle.json" in uploaded:
            uploaded_kaggle_json = Path("kaggle.json")
        else:
            raise RuntimeError("Upload kaggle.json, or set KAGGLE_USERNAME and KAGGLE_KEY in Colab secrets.")

from sleep_stage.kaggle_download import configure_kaggle_credentials

credential_path = configure_kaggle_credentials(uploaded_kaggle_json)
if credential_path is None:
    raise RuntimeError("Kaggle credentials are not configured.")
print("Kaggle credentials configured.")
```

Cell 3: download and extract the competition data before reading files.

```python
from sleep_stage.config import COMPETITION_SLUG
from sleep_stage.kaggle_download import download_and_extract_competition

competition_root = download_and_extract_competition(INPUT_ROOT, COMPETITION_SLUG)
print(f"Competition data ready: {competition_root}")
```

Cell 4: load data, build features, and run grouped-validation experiments.

```python
from sleep_stage.config import build_paths
from sleep_stage.pipeline import load_or_build_features, run_experiments

paths = build_paths(
    project_root=PROJECT_ROOT,
    competition_root=competition_root,
    output_path=SUBMISSION_PATH,
    working_dir=WORKING_DIR,
)
train_table, test_table = load_or_build_features(paths, force=False)
print("train_table", train_table.shape)
print("test_table", test_table.shape)
print(train_table["label"].value_counts().to_dict())

results = run_experiments(paths=paths, force_features=False)
print(results[["experiment", "model_name", "n_features", "weighted_f1_mean", "weighted_f1_std"]].to_string(index=False))
```

Cell 5: generate and validate `/content/submission.csv`.

```python
import pandas as pd

from sleep_stage.data import validate_submission
from sleep_stage.pipeline import generate_submission

submission_path = generate_submission(paths)
validation = validate_submission(submission_path, paths.sample_submission)
submission = pd.read_csv(submission_path)

print(validation)
print(submission.head().to_string(index=False))
print(submission["labels"].value_counts().to_string())
print(f"Submission CSV ready: {submission_path}")
assert submission_path == Path("/content/submission.csv") or LOCAL_ROOT.exists()
assert len(submission) == 7832
```

- [ ] **Step 2: Generate the `.ipynb` file using nbformat**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python - <<'PY'
from pathlib import Path
import nbformat as nbf

cell_1 = Path("notebook_cells/cell_1.py").read_text()
cell_2 = Path("notebook_cells/cell_2.py").read_text()
cell_3 = Path("notebook_cells/cell_3.py").read_text()
cell_4 = Path("notebook_cells/cell_4.py").read_text()
cell_5 = Path("notebook_cells/cell_5.py").read_text()

notebook = nbf.v4.new_notebook()
notebook["cells"] = [nbf.v4.new_code_cell(cell) for cell in [cell_1, cell_2, cell_3, cell_4, cell_5]]
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
nbf.write(notebook, Path("sleep_stage_solution.ipynb"))
PY
```

Expected output:

```text
```

If the implementation does not create a temporary `notebook_cells/` directory, generate the notebook by assigning the exact five cell sources from Step 1 to `cell_1` through `cell_5` in the same script. Do not leave placeholder cell text in the saved notebook.

- [ ] **Step 3: Verify the notebook downloads but does not submit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python - <<'PY'
from pathlib import Path

text = Path("sleep_stage_solution.ipynb").read_text()
required = ["competitions download", "/content/input", "/content/submission.csv", "kaggle.json"]
blocked = [
    "kaggle competitions " + "submit",
    "." + "submit(",
    "competition_" + "submit",
    "competitions " + "submit",
]
missing = [token for token in required if token not in text]
found = [token for token in blocked if token in text]
print("missing required tokens:", missing)
print("blocked tokens found:", found)
assert not missing
assert not found
PY
```

Expected output:

```text
missing required tokens: []
blocked tokens found: []
```

## Task 11: Submission Generation And Validation

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/working/submission.csv`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/submission_log.md`

- [ ] **Step 1: Generate local fallback submission**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python - <<'PY'
from sleep_stage.pipeline import generate_submission

path = generate_submission()
print(path)
PY
```

Expected output:

```text
Validated submission: rows=7832, labels=['N1', 'N2', 'N3', 'R', 'W']
/Users/temicide/Documents/5_domain_final/Sleep-Stage/working/submission.csv
```

- [ ] **Step 2: Validate format and row count**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python - <<'PY'
import pandas as pd
from sleep_stage.config import build_paths
from sleep_stage.data import validate_submission

paths = build_paths()
validation = validate_submission(paths.output_path, paths.sample_submission)
submission = pd.read_csv(paths.output_path)
print(validation)
print(submission.head(3).to_string(index=False))
print(submission.tail(3).to_string(index=False))
print(submission["labels"].value_counts().to_string())
PY
```

Expected output:

```text
SubmissionValidation(rows=7832,
          id labels
test001_00000
test001_00001
test001_00002
          id labels
test010_00780
test010_00781
test010_00782
N2
W
N1
R
N3
```

Each printed label must be one of `W`, `N1`, `N2`, `N3`, or `R`; the validator raises an exception before this output if any ID or label is invalid.

- [ ] **Step 3: Record the manual submission log**

```markdown
# /Users/temicide/Documents/5_domain_final/Sleep-Stage/submission_log.md
# Sleep-Stage Submission Log

| Submission | Local CV | Public LB | Change |
| --- | ---: | ---: | --- |
| baseline format check | 0.34142 | not uploaded | N2-only sanity file generated only to verify CSV shape |
```

When a real Kaggle candidate is produced, append one concrete row after recording the local grouped CV score and the manually observed public leaderboard score. Use this exact format:

```markdown
| context HGB | 0.53821 | 0.60114 | Previous/next and rolling context features |
```

Do not add any competition submission command to any file.

## Task 12: Self-Review And Acceptance Checks

**Files:**
- Inspect: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/spec.md`
- Inspect: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/plan.md`
- Inspect: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/*.py`
- Inspect: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/sleep_stage_solution.ipynb`

- [ ] **Step 1: Run placeholder scan**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python - <<'PY'
from pathlib import Path

patterns = ["TB" + "D", "TO" + "DO", "implement " + "later", "appro" + "priate", "similar " + "to"]
targets = [Path("plan.md"), *Path("src").glob("sleep_stage/*.py"), Path("sleep_stage_solution.ipynb")]
for target in targets:
    if not target.exists():
        continue
    text = target.read_text()
    hits = [pattern for pattern in patterns if pattern in text]
    print(target, hits)
    assert not hits, f"{target} contains placeholder wording: {hits}"
PY
```

Expected output:

```text
plan.md []
src/sleep_stage/__init__.py []
src/sleep_stage/config.py []
src/sleep_stage/kaggle_download.py []
src/sleep_stage/data.py []
src/sleep_stage/features.py []
src/sleep_stage/context.py []
src/sleep_stage/models.py []
src/sleep_stage/smoothing.py []
src/sleep_stage/pipeline.py []
sleep_stage_solution.ipynb []
```

- [ ] **Step 2: Run full local tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest -v
```

Expected output:

```text
13 passed
```

- [ ] **Step 3: Verify download/extract is present and automated submission is absent**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
rg -n "competitions download|competition_download_files|/content/input|/content/submission.csv|kaggle.json" spec.md plan.md src sleep_stage_solution.ipynb
python - <<'PY'
from pathlib import Path

patterns = [
    "kaggle competitions " + "submit",
    "competitions " + "submit",
    "competition_" + "submit",
    "." + "submit(",
]
targets = [Path("spec.md"), Path("plan.md"), *Path("src").glob("sleep_stage/*.py"), Path("sleep_stage_solution.ipynb")]
for target in targets:
    if target.exists():
        hits = [pattern for pattern in patterns if pattern in target.read_text()]
        print(target, hits)
        assert not hits
PY
```

Expected output:

```text
# First command prints required download/path/credential references.
# Second command prints each inspected file with an empty hit list.
```

- [ ] **Step 4: Verify final submission file**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python - <<'PY'
from sleep_stage.config import build_paths
from sleep_stage.data import validate_submission

paths = build_paths()
validation = validate_submission(paths.output_path, paths.sample_submission)
print(validation.rows)
print(validation.id_match)
print(sorted(set(validation.valid_labels)))
print(paths.output_path)
PY
```

Expected output:

```text
7832
True
['N1', 'N2', 'N3', 'R', 'W']
/Users/temicide/Documents/5_domain_final/Sleep-Stage/working/submission.csv
```

- [ ] **Step 5: Review spec coverage**

Confirm the following mapping before declaring completion:

```text
Colab end-to-end: Task 10
Kaggle CLI/API download before reads: Task 2 and Task 10
Credential handling without printing secrets: Task 2 and Task 10
Archive extraction before loading files: Task 2 and Task 10
/content/input extracted data path: Task 1, Task 2, and Task 10
/content/working cache path: Task 1, Task 8, and Task 10
/content/submission.csv generation: Task 1, Task 8, Task 10, and Task 11
Local development fallback: Task 1 and Task 9
CSV validation before completion: Task 1, Task 8, Task 10, Task 11, and Task 12
No automated competition submission: Task 10 and Task 12
480-row epoch parsing: Task 3
Grouped validation by recording: Task 6 and Task 9
Weighted F1 and per-class F1 reporting: Task 6 and Task 9
Rich epoch features: Task 4
Neighbor and rolling context: Task 5
Temporal smoothing and Viterbi decoding: Task 7
Manual submission strategy and log: Task 11
```

## Self-Review Notes

**Spec coverage:** This plan covers Colab execution, Kaggle CLI/API download, credential handling via uploaded `kaggle.json` or environment variables without printing secrets, archive extraction before file reads, Colab-first paths with local fallbacks, `/content/submission.csv` generation, submission validation, no automated competition submission, 480-row epoch parsing, grouped validation by full recording, weighted and per-class F1 reporting, rich epoch features, neighbor and rolling context features, Viterbi smoothing, and a manual submission log.

**Placeholder scan:** The plan avoids unresolved implementation placeholders in requirements and acceptance checks. The notebook generation step requires the saved notebook to contain the concrete five cells above, not placeholder cell text.

**Path consistency:** Shared names are consistent across tasks: `COLAB_INPUT_ROOT`, `COLAB_WORKING_DIR`, `COLAB_OUTPUT_PATH`, `ProjectPaths.output_path`, `ProjectPaths.working_dir`, `COMPETITION_SLUG`, `EPOCH_ROWS`, `SIGNAL_COLUMNS`, `LABELS`, `SubmissionValidation`, `download_and_extract_competition`, `verify_extracted_competition_data`, `add_context_features`, `viterbi_decode`, and `generate_submission`.

## Execution Handoff

Plan complete and saved to `/Users/temicide/Documents/5_domain_final/Sleep-Stage/plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.
