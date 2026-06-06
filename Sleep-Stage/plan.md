# Sleep-Stage Kaggle Solution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Kaggle Notebook solution that trains grouped-validation sleep-stage models, generates a validated `/kaggle/working/submission.csv`, and avoids automated Kaggle submission.

**Architecture:** Use a focused Python package under `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage` for data loading, feature engineering, grouped validation, context features, smoothing, and submission generation. Keep `/Users/temicide/Documents/5_domain_final/Sleep-Stage/sleep_stage_solution.ipynb` as the Kaggle-ready executable wrapper that calls the same package logic, with local path fallbacks for development. Model selection is driven by `GroupKFold` over full train recordings, then the final model predicts test segment IDs from `sample_submission.csv`.

**Tech Stack:** Python 3.10+, NumPy, pandas, scikit-learn, SciPy, joblib, pytest, optional LightGBM when already available in the runtime.

---

## File Structure

- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/pyproject.toml`: local package metadata, pytest configuration, and runtime dependencies.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/__init__.py`: package marker and version string.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/config.py`: constants for labels, paths, sampling rate, epoch length, and Kaggle/local root discovery.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/data.py`: train recording loading, test segment loading, epoch splitting, sample-submission mapping, and CSV submission validation.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/features.py`: deterministic per-epoch feature extraction for time statistics, motion, HR/IBI, BVP spectrum, and cross-signal features.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/context.py`: previous/next epoch context, rolling statistics, epoch deltas, and relative-position features.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/models.py`: model factories, grouped CV runner, fold reports, final fit, and probability blending.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/smoothing.py`: mode filters and Viterbi decoding from train transition counts.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/pipeline.py`: command-style orchestration for cache building, validation experiments, and submission generation.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_data.py`: synthetic tests for epoch splitting, path discovery, and submission validation.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_features_context.py`: synthetic tests for feature names, finite values, and context feature shape.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_models_smoothing.py`: synthetic tests for grouped validation plumbing and smoothing behavior.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/sleep_stage_solution.ipynb`: Kaggle Notebook-ready solution that installs no new packages, reads `/kaggle/input/...` on Kaggle, uses local fallback during development, trains the best validated pipeline, validates predictions, and writes `/kaggle/working/submission.csv`.
- Create `/Users/temicide/Documents/5_domain_final/Sleep-Stage/submission_log.md`: manual record of local CV, public LB, and change description for each of the five planned submissions.

## Task 1: Package Skeleton And Constants

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/pyproject.toml`
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/__init__.py`
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/config.py`
- Test: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

```python
# /Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_data.py
from pathlib import Path

import pandas as pd

from sleep_stage.config import (
    EPOCH_ROWS,
    LABELS,
    SIGNAL_COLUMNS,
    find_competition_root,
)
from sleep_stage.data import validate_submission


def test_constants_match_competition_spec():
    assert EPOCH_ROWS == 480
    assert SIGNAL_COLUMNS == ["BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA", "HR", "IBI"]
    assert LABELS == ["W", "N1", "N2", "N3", "R"]


def test_find_competition_root_prefers_kaggle_then_local(tmp_path: Path):
    kaggle_root = tmp_path / "kaggle" / "input" / "super-ai-engineer-ss-6-individual-sleep-stage-classification"
    local_root = tmp_path / "local" / "data" / "super-ai-engineer-ss-6-individual-sleep-stage-classification"
    kaggle_root.mkdir(parents=True)
    local_root.mkdir(parents=True)

    found = find_competition_root(
        kaggle_input_root=tmp_path / "kaggle" / "input",
        local_data_root=tmp_path / "local" / "data",
    )

    assert found == kaggle_root


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


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    competition_root: Path
    train_dir: Path
    test_dir: Path
    sample_submission: Path
    cache_dir: Path
    output_dir: Path


def find_competition_root(
    kaggle_input_root: Path = Path("/kaggle/input"),
    local_data_root: Path | None = None,
) -> Path:
    local_root = local_data_root or Path(__file__).resolve().parents[2] / "data"
    candidates = [
        kaggle_input_root / COMPETITION_SLUG,
        local_root / COMPETITION_SLUG,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Competition data not found. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def build_paths(
    project_root: Path | None = None,
    output_dir: Path | None = None,
    competition_root: Path | None = None,
) -> ProjectPaths:
    root = project_root or Path(__file__).resolve().parents[2]
    data_root = competition_root or find_competition_root(local_data_root=root / "data")
    out = output_dir or (Path("/kaggle/working") if Path("/kaggle/working").exists() else root / "working")
    return ProjectPaths(
        project_root=root,
        competition_root=data_root,
        train_dir=data_root / "train" / "train",
        test_dir=data_root / "test_segment" / "test_segment",
        sample_submission=data_root / "sample_submission.csv",
        cache_dir=root / "cache",
        output_dir=out,
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
tests/test_data.py::test_find_competition_root_prefers_kaggle_then_local PASSED
tests/test_data.py::test_validate_submission_accepts_complete_known_labels PASSED
```

- [ ] **Step 6: Commit**

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
git add pyproject.toml src/sleep_stage/__init__.py src/sleep_stage/config.py src/sleep_stage/data.py tests/test_data.py
git commit -m "chore: scaffold sleep stage package"
```

Expected output contains:

```text
chore: scaffold sleep stage package
 5 files changed
```

## Task 2: Data Loading And Epoch Splitting

**Files:**
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/data.py`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_data.py`

- [ ] **Step 1: Write failing tests for epoch conversion and test ID mapping**

```python
# Append to /Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_data.py
import numpy as np

from sleep_stage.data import (
    epoch_dataframe_from_recording,
    list_test_segment_paths,
    read_test_segments,
)


def _synthetic_recording(label_a: str = "W", label_b: str = "N2") -> pd.DataFrame:
    rows = 960
    frame = pd.DataFrame(
        {
            "BVP": np.linspace(0.0, 1.0, rows),
            "ACC_X": np.zeros(rows),
            "ACC_Y": np.ones(rows),
            "ACC_Z": np.full(rows, 2.0),
            "TEMP": np.full(rows, 33.0),
            "EDA": np.full(rows, 0.2),
            "HR": np.full(rows, 70.0),
            "IBI": np.full(rows, 0.85),
            "Sleep_Stage": [label_a] * 480 + [label_b] * 480,
        }
    )
    return frame


def test_epoch_dataframe_from_recording_returns_one_row_per_480_samples():
    frame = _synthetic_recording()

    epochs = epoch_dataframe_from_recording(frame, recording_id="train_001")

    assert epochs[["recording_id", "epoch_index", "label"]].to_dict("records") == [
        {"recording_id": "train_001", "epoch_index": 0, "label": "W"},
        {"recording_id": "train_001", "epoch_index": 1, "label": "N2"},
    ]
    assert epochs.loc[0, "start_row"] == 0
    assert epochs.loc[1, "start_row"] == 480
    assert epochs.loc[1, "end_row"] == 960


def test_list_and_read_test_segments_follow_sample_submission_order(tmp_path: Path):
    test_root = tmp_path / "test_segment" / "test_segment"
    subject_dir = test_root / "test001"
    subject_dir.mkdir(parents=True)
    pd.DataFrame({"id": ["test001_00001", "test001_00000"], "labels": ["N2", "N2"]}).to_csv(
        tmp_path / "sample_submission.csv",
        index=False,
    )
    for segment_id in ["test001_00000", "test001_00001"]:
        segment = _synthetic_recording().drop(columns=["Sleep_Stage"]).iloc[:480]
        segment.to_csv(subject_dir / f"{segment_id}.csv", index=False)

    paths = list_test_segment_paths(test_root)
    segments = read_test_segments(tmp_path / "sample_submission.csv", paths)

    assert list(paths) == ["test001_00000", "test001_00001"]
    assert list(segments) == ["test001_00001", "test001_00000"]
    assert segments["test001_00000"].shape == (480, 8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_data.py -v
```

Expected output:

```text
E   ImportError: cannot import name 'epoch_dataframe_from_recording'
```

- [ ] **Step 3: Add loading and epoch helpers**

```python
# Add to /Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/data.py
from collections.abc import Mapping

import numpy as np

from sleep_stage.config import EPOCH_ROWS, LABEL_COLUMN, SIGNAL_COLUMNS


def _validate_signal_columns(frame: pd.DataFrame, require_label: bool) -> None:
    required = SIGNAL_COLUMNS + ([LABEL_COLUMN] if require_label else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def epoch_dataframe_from_recording(frame: pd.DataFrame, recording_id: str) -> pd.DataFrame:
    _validate_signal_columns(frame, require_label=True)
    usable_rows = (len(frame) // EPOCH_ROWS) * EPOCH_ROWS
    if usable_rows == 0:
        raise ValueError(f"recording {recording_id} has {len(frame)} rows; at least {EPOCH_ROWS} rows required")
    if usable_rows != len(frame):
        frame = frame.iloc[:usable_rows].copy()
    records: list[dict[str, object]] = []
    for epoch_index, start in enumerate(range(0, usable_rows, EPOCH_ROWS)):
        end = start + EPOCH_ROWS
        labels = frame[LABEL_COLUMN].iloc[start:end].astype(str)
        label_counts = labels.value_counts()
        label = str(label_counts.index[0])
        if len(label_counts) > 1:
            raise ValueError(f"recording {recording_id} epoch {epoch_index} has non-constant labels {label_counts.to_dict()}")
        records.append(
            {
                "recording_id": recording_id,
                "epoch_index": epoch_index,
                "start_row": start,
                "end_row": end,
                "label": label,
            }
        )
    return pd.DataFrame.from_records(records)


def load_train_recording(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _validate_signal_columns(frame, require_label=True)
    return frame


def iter_train_recordings(train_dir: Path) -> list[tuple[str, Path]]:
    paths = sorted(train_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no train CSV files found in {train_dir}")
    return [(path.stem, path) for path in paths]


def list_test_segment_paths(test_root: Path) -> dict[str, Path]:
    paths = {path.stem: path for path in sorted(test_root.glob("test*/*.csv"))}
    if not paths:
        raise FileNotFoundError(f"no test segment CSV files found in {test_root}")
    return paths


def read_test_segments(sample_submission_path: Path, segment_paths: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    sample = pd.read_csv(sample_submission_path)
    segments: dict[str, pd.DataFrame] = {}
    for segment_id in sample["id"].astype(str):
        if segment_id not in segment_paths:
            raise FileNotFoundError(f"missing test segment CSV for id {segment_id}")
        frame = pd.read_csv(segment_paths[segment_id])
        _validate_signal_columns(frame, require_label=False)
        if len(frame) != EPOCH_ROWS:
            raise ValueError(f"test segment {segment_id} has {len(frame)} rows; expected {EPOCH_ROWS}")
        segments[segment_id] = frame[SIGNAL_COLUMNS].astype(np.float32)
    return segments
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_data.py -v
```

Expected output:

```text
tests/test_data.py::test_epoch_dataframe_from_recording_returns_one_row_per_480_samples PASSED
tests/test_data.py::test_list_and_read_test_segments_follow_sample_submission_order PASSED
```

- [ ] **Step 5: Commit**

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
git add src/sleep_stage/data.py tests/test_data.py
git commit -m "feat: load sleep stage epochs"
```

Expected output contains:

```text
feat: load sleep stage epochs
 2 files changed
```

## Task 3: Per-Epoch Feature Extraction

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/features.py`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_features_context.py`

- [ ] **Step 1: Write failing feature tests**

```python
# /Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_features_context.py
import numpy as np
import pandas as pd

from sleep_stage.features import extract_epoch_features, extract_feature_table


SIGNALS = ["BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA", "HR", "IBI"]


def _epoch_frame(scale: float = 1.0) -> pd.DataFrame:
    x = np.linspace(0.0, scale, 480, dtype=np.float32)
    return pd.DataFrame(
        {
            "BVP": np.sin(np.linspace(0.0, 4.0 * np.pi, 480)) * scale,
            "ACC_X": x,
            "ACC_Y": x * 0.5,
            "ACC_Z": x * -0.25,
            "TEMP": 33.0 + x * 0.01,
            "EDA": 0.2 + x * 0.02,
            "HR": 70.0 + x,
            "IBI": 0.85 - x * 0.001,
        }
    )


def test_extract_epoch_features_has_deterministic_finite_values():
    features = extract_epoch_features(_epoch_frame())

    assert "BVP_mean" in features
    assert "BVP_fft_peak_hz" in features
    assert "ACC_mag_mean" in features
    assert "HR_IBI_consistency" in features
    assert all(np.isfinite(value) for value in features.values())
    assert list(features) == sorted(features)


def test_extract_feature_table_returns_metadata_and_sorted_feature_columns():
    table = extract_feature_table(
        [
            {"recording_id": "rec_a", "epoch_index": 0, "label": "W", "frame": _epoch_frame(1.0)},
            {"recording_id": "rec_a", "epoch_index": 1, "label": "N2", "frame": _epoch_frame(2.0)},
        ]
    )

    assert table[["recording_id", "epoch_index", "label"]].to_dict("records") == [
        {"recording_id": "rec_a", "epoch_index": 0, "label": "W"},
        {"recording_id": "rec_a", "epoch_index": 1, "label": "N2"},
    ]
    feature_columns = [column for column in table.columns if column not in {"recording_id", "epoch_index", "label"}]
    assert feature_columns == sorted(feature_columns)
    assert table[feature_columns].notna().all().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_features_context.py -v
```

Expected output:

```text
E   ModuleNotFoundError: No module named 'sleep_stage.features'
```

- [ ] **Step 3: Implement deterministic feature extraction**

```python
# /Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/features.py
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from sleep_stage.config import SAMPLING_HZ, SIGNAL_COLUMNS


def _safe_float(value: float) -> float:
    result = float(value)
    if np.isfinite(result):
        return result
    return 0.0


def _slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float32)
    centered_x = x - x.mean()
    centered_y = values - values.mean()
    denom = float(np.sum(centered_x * centered_x))
    if denom == 0.0:
        return 0.0
    return _safe_float(np.sum(centered_x * centered_y) / denom)


def _channel_stats(name: str, values: np.ndarray) -> dict[str, float]:
    diff = np.diff(values)
    quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    centered = values - np.median(values)
    return {
        f"{name}_diff_abs_max": _safe_float(np.max(np.abs(diff))) if diff.size else 0.0,
        f"{name}_diff_abs_mean": _safe_float(np.mean(np.abs(diff))) if diff.size else 0.0,
        f"{name}_iqr": _safe_float(quantiles[3] - quantiles[1]),
        f"{name}_kurtosis": _safe_float(stats.kurtosis(values, fisher=True, bias=False)),
        f"{name}_max": _safe_float(np.max(values)),
        f"{name}_mean": _safe_float(np.mean(values)),
        f"{name}_median_abs_dev": _safe_float(np.median(np.abs(centered))),
        f"{name}_min": _safe_float(np.min(values)),
        f"{name}_p10": _safe_float(quantiles[0]),
        f"{name}_p25": _safe_float(quantiles[1]),
        f"{name}_p50": _safe_float(quantiles[2]),
        f"{name}_p75": _safe_float(quantiles[3]),
        f"{name}_p90": _safe_float(quantiles[4]),
        f"{name}_range": _safe_float(np.max(values) - np.min(values)),
        f"{name}_skew": _safe_float(stats.skew(values, bias=False)),
        f"{name}_slope": _safe_float(_slope(values)),
        f"{name}_std": _safe_float(np.std(values)),
        f"{name}_zero_cross_proxy": _safe_float(np.mean(np.diff(np.signbit(values - np.median(values))) != 0)),
    }


def _bvp_frequency_features(values: np.ndarray) -> dict[str, float]:
    centered = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(values.size, d=1.0 / SAMPLING_HZ)
    total_power = float(np.sum(spectrum) + 1e-9)
    bands = {
        "BVP_fft_0p04_0p15": (0.04, 0.15),
        "BVP_fft_0p15_0p40": (0.15, 0.40),
        "BVP_fft_0p40_1p00": (0.40, 1.00),
        "BVP_fft_1p00_3p00": (1.00, 3.00),
    }
    output = {}
    for key, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        output[f"{key}_ratio"] = _safe_float(np.sum(spectrum[mask]) / total_power)
    nonzero = freqs > 0
    if np.any(nonzero):
        peak_index = np.argmax(spectrum[nonzero])
        output["BVP_fft_peak_hz"] = _safe_float(freqs[nonzero][peak_index])
    else:
        output["BVP_fft_peak_hz"] = 0.0
    return output


def extract_epoch_features(frame: pd.DataFrame) -> dict[str, float]:
    values = frame[SIGNAL_COLUMNS].astype(np.float32)
    features: dict[str, float] = {}
    for column in SIGNAL_COLUMNS:
        series = values[column].to_numpy(dtype=np.float32)
        features.update(_channel_stats(column, series))
    acc = values[["ACC_X", "ACC_Y", "ACC_Z"]].to_numpy(dtype=np.float32)
    acc_mag = np.sqrt(np.sum(acc * acc, axis=1))
    acc_jerk = np.sqrt(np.sum(np.diff(acc, axis=0) ** 2, axis=1))
    features.update(_channel_stats("ACC_mag", acc_mag))
    features.update(
        {
            "ACC_jerk_max": _safe_float(np.max(acc_jerk)) if acc_jerk.size else 0.0,
            "ACC_jerk_mean": _safe_float(np.mean(acc_jerk)) if acc_jerk.size else 0.0,
            "ACC_jerk_std": _safe_float(np.std(acc_jerk)) if acc_jerk.size else 0.0,
        }
    )
    features.update(_bvp_frequency_features(values["BVP"].to_numpy(dtype=np.float32)))
    hr = values["HR"].to_numpy(dtype=np.float32)
    ibi = values["IBI"].to_numpy(dtype=np.float32)
    features["HR_rmssd"] = _safe_float(np.sqrt(np.mean(np.diff(hr) ** 2))) if hr.size > 1 else 0.0
    features["IBI_rmssd"] = _safe_float(np.sqrt(np.mean(np.diff(ibi) ** 2))) if ibi.size > 1 else 0.0
    features["HR_IBI_consistency"] = _safe_float(np.mean(hr) * np.mean(ibi))
    features["ACC_mag_x_HR"] = _safe_float(np.mean(acc_mag) * np.mean(hr))
    features["EDA_x_TEMP"] = _safe_float(np.mean(values["EDA"]) * np.mean(values["TEMP"]))
    return {key: features[key] for key in sorted(features)}


def extract_feature_table(epoch_items: Iterable[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in epoch_items:
        row = {
            "recording_id": str(item["recording_id"]),
            "epoch_index": int(item["epoch_index"]),
        }
        if "label" in item:
            row["label"] = str(item["label"])
        row.update(extract_epoch_features(item["frame"]))
        rows.append(row)
    table = pd.DataFrame.from_records(rows)
    metadata = [column for column in ["recording_id", "segment_id", "epoch_index", "label"] if column in table.columns]
    feature_columns = sorted(column for column in table.columns if column not in metadata)
    return table[metadata + feature_columns]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_features_context.py -v
```

Expected output:

```text
tests/test_features_context.py::test_extract_epoch_features_has_deterministic_finite_values PASSED
tests/test_features_context.py::test_extract_feature_table_returns_metadata_and_sorted_feature_columns PASSED
```

- [ ] **Step 5: Commit**

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
git add src/sleep_stage/features.py tests/test_features_context.py
git commit -m "feat: extract epoch features"
```

Expected output contains:

```text
feat: extract epoch features
 2 files changed
```

## Task 4: Context And Rolling Features

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/context.py`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_features_context.py`

- [ ] **Step 1: Write failing context tests**

```python
# Append to /Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_features_context.py
from sleep_stage.context import add_context_features


def test_add_context_features_preserves_rows_and_adds_temporal_columns():
    table = pd.DataFrame(
        {
            "recording_id": ["rec_a", "rec_a", "rec_a", "rec_b"],
            "epoch_index": [0, 1, 2, 0],
            "label": ["W", "N1", "N2", "R"],
            "BVP_mean": [1.0, 2.0, 4.0, 10.0],
            "HR_mean": [60.0, 62.0, 64.0, 70.0],
        }
    )

    context = add_context_features(table, lags=(1, 2), rolling_windows=(3,))

    assert len(context) == 4
    assert context.loc[0, "relative_position"] == 0.0
    assert context.loc[2, "relative_position"] == 1.0
    assert context.loc[1, "BVP_mean_prev_1"] == 1.0
    assert context.loc[1, "BVP_mean_next_1"] == 4.0
    assert context.loc[0, "BVP_mean_prev_1"] == 1.0
    assert context.loc[3, "BVP_mean_next_1"] == 10.0
    assert "HR_mean_roll3_mean" in context.columns
    assert "BVP_mean_delta_prev" in context.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_features_context.py::test_add_context_features_preserves_rows_and_adds_temporal_columns -v
```

Expected output:

```text
E   ModuleNotFoundError: No module named 'sleep_stage.context'
```

- [ ] **Step 3: Implement temporal context features**

```python
# /Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/context.py
from __future__ import annotations

import pandas as pd


METADATA_COLUMNS = {"recording_id", "segment_id", "epoch_index", "label"}


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in table.columns if column not in METADATA_COLUMNS]


def add_context_features(
    table: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3),
    rolling_windows: tuple[int, ...] = (3, 5, 9, 15),
) -> pd.DataFrame:
    ordered = table.sort_values(["recording_id", "epoch_index"]).reset_index(drop=True).copy()
    base_features = feature_columns(ordered)
    grouped = ordered.groupby("recording_id", sort=False)
    n_epochs = grouped["epoch_index"].transform("count").astype(float)
    epoch_rank = grouped.cumcount().astype(float)
    ordered["relative_position"] = (epoch_rank / (n_epochs - 1).clip(lower=1)).fillna(0.0)
    for column in base_features:
        for lag in lags:
            previous = grouped[column].shift(lag)
            following = grouped[column].shift(-lag)
            ordered[f"{column}_prev_{lag}"] = previous.fillna(ordered[column])
            ordered[f"{column}_next_{lag}"] = following.fillna(ordered[column])
        ordered[f"{column}_delta_prev"] = (ordered[column] - grouped[column].shift(1)).fillna(0.0)
        ordered[f"{column}_delta_next"] = (grouped[column].shift(-1) - ordered[column]).fillna(0.0)
        for window in rolling_windows:
            rolled = grouped[column].rolling(window=window, center=True, min_periods=1)
            ordered[f"{column}_roll{window}_mean"] = rolled.mean().reset_index(level=0, drop=True)
            ordered[f"{column}_roll{window}_std"] = rolled.std().reset_index(level=0, drop=True).fillna(0.0)
    metadata = [column for column in ["recording_id", "segment_id", "epoch_index", "label"] if column in ordered.columns]
    features = sorted(column for column in ordered.columns if column not in metadata)
    return ordered[metadata + features]
```

- [ ] **Step 4: Run context tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_features_context.py -v
```

Expected output:

```text
tests/test_features_context.py::test_add_context_features_preserves_rows_and_adds_temporal_columns PASSED
```

- [ ] **Step 5: Commit**

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
git add src/sleep_stage/context.py tests/test_features_context.py
git commit -m "feat: add temporal context features"
```

Expected output contains:

```text
feat: add temporal context features
 2 files changed
```

## Task 5: Grouped Cross-Validation Models

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/models.py`
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_models_smoothing.py`

- [ ] **Step 1: Write failing grouped-CV tests**

```python
# /Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_models_smoothing.py
import numpy as np
import pandas as pd

from sleep_stage.models import build_model, run_grouped_cv


def _classification_table() -> pd.DataFrame:
    rows = []
    labels = ["W", "N1", "N2", "N3", "R"]
    for group_index in range(10):
        label = labels[group_index % len(labels)]
        for epoch_index in range(6):
            rows.append(
                {
                    "recording_id": f"rec_{group_index:02d}",
                    "epoch_index": epoch_index,
                    "label": label,
                    "feature_a": float(group_index),
                    "feature_b": float(epoch_index),
                }
            )
    return pd.DataFrame(rows)


def test_build_model_returns_predict_proba_estimator():
    model = build_model("extra_trees", random_state=7)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict_proba")


def test_run_grouped_cv_returns_fold_metrics_and_oof_probabilities():
    report = run_grouped_cv(
        _classification_table(),
        model_name="extra_trees",
        n_splits=5,
        random_state=7,
    )

    assert len(report.fold_scores) == 5
    assert set(report.per_class_f1.columns) == {"fold", "W", "N1", "N2", "N3", "R"}
    assert report.oof_probabilities.shape == (60, 5)
    assert np.allclose(report.oof_probabilities.sum(axis=1), 1.0)
    assert 0.0 <= report.weighted_f1_mean <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_models_smoothing.py -v
```

Expected output:

```text
E   ModuleNotFoundError: No module named 'sleep_stage.models'
```

- [ ] **Step 3: Implement model factories and grouped validation**

```python
# /Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/models.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from sleep_stage.config import LABELS, LABEL_TO_INDEX
from sleep_stage.context import METADATA_COLUMNS


@dataclass(frozen=True)
class CVReport:
    fold_scores: list[float]
    weighted_f1_mean: float
    weighted_f1_std: float
    per_class_f1: pd.DataFrame
    confusion: np.ndarray
    oof_probabilities: np.ndarray
    feature_columns: list[str]


def get_feature_columns(table: pd.DataFrame) -> list[str]:
    return sorted(column for column in table.columns if column not in METADATA_COLUMNS)


def encode_labels(labels: pd.Series) -> np.ndarray:
    return labels.astype(str).map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)


def build_model(model_name: str, random_state: int = 42):
    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=420,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=random_state,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=550,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=450,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"unknown model_name {model_name}; expected hgb, extra_trees, or random_forest")


def _aligned_predict_proba(model, x_valid: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x_valid)
    output = np.zeros((len(x_valid), len(LABELS)), dtype=np.float32)
    for local_index, class_index in enumerate(model.classes_):
        output[:, int(class_index)] = raw[:, local_index]
    row_sums = output.sum(axis=1, keepdims=True)
    return output / np.clip(row_sums, 1e-9, None)


def run_grouped_cv(
    table: pd.DataFrame,
    model_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> CVReport:
    feature_cols = get_feature_columns(table)
    x = table[feature_cols].astype(np.float32)
    y = encode_labels(table["label"])
    groups = table["recording_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.zeros((len(table), len(LABELS)), dtype=np.float32)
    fold_scores: list[float] = []
    per_class_rows: list[dict[str, float | int]] = []
    total_confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups), start=1):
        model = build_model(model_name, random_state=random_state + fold)
        model.fit(x.iloc[train_idx], y[train_idx])
        probabilities = _aligned_predict_proba(model, x.iloc[valid_idx])
        predictions = probabilities.argmax(axis=1)
        oof[valid_idx] = probabilities
        score = f1_score(y[valid_idx], predictions, average="weighted", labels=list(range(len(LABELS))))
        fold_scores.append(float(score))
        report = classification_report(
            y[valid_idx],
            predictions,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        )
        row: dict[str, float | int] = {"fold": fold}
        for label in LABELS:
            row[label] = float(report[label]["f1-score"])
        per_class_rows.append(row)
        total_confusion += confusion_matrix(y[valid_idx], predictions, labels=list(range(len(LABELS))))
    return CVReport(
        fold_scores=fold_scores,
        weighted_f1_mean=float(np.mean(fold_scores)),
        weighted_f1_std=float(np.std(fold_scores)),
        per_class_f1=pd.DataFrame(per_class_rows),
        confusion=total_confusion,
        oof_probabilities=oof,
        feature_columns=feature_cols,
    )


def fit_final_model(table: pd.DataFrame, model_name: str, random_state: int = 42):
    feature_cols = get_feature_columns(table)
    model = build_model(model_name, random_state=random_state)
    model.fit(table[feature_cols].astype(np.float32), encode_labels(table["label"]))
    return model, feature_cols


def blend_probabilities(probability_sets: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if len(probability_sets) != len(weights):
        raise ValueError("probability_sets and weights must have the same length")
    weights_array = np.asarray(weights, dtype=np.float32)
    weights_array = weights_array / np.sum(weights_array)
    blended = np.zeros_like(probability_sets[0], dtype=np.float32)
    for probabilities, weight in zip(probability_sets, weights_array, strict=True):
        blended += probabilities.astype(np.float32) * float(weight)
    return blended / np.clip(blended.sum(axis=1, keepdims=True), 1e-9, None)
```

- [ ] **Step 4: Run model tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_models_smoothing.py -v
```

Expected output:

```text
tests/test_models_smoothing.py::test_build_model_returns_predict_proba_estimator PASSED
tests/test_models_smoothing.py::test_run_grouped_cv_returns_fold_metrics_and_oof_probabilities PASSED
```

- [ ] **Step 5: Commit**

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
git add src/sleep_stage/models.py tests/test_models_smoothing.py
git commit -m "feat: evaluate grouped models"
```

Expected output contains:

```text
feat: evaluate grouped models
 2 files changed
```

## Task 6: Temporal Smoothing And Viterbi Decoding

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/smoothing.py`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_models_smoothing.py`

- [ ] **Step 1: Write failing smoothing tests**

```python
# Append to /Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_models_smoothing.py
from sleep_stage.config import LABEL_TO_INDEX
from sleep_stage.smoothing import build_transition_log_probs, mode_filter_labels, viterbi_decode


def test_mode_filter_labels_removes_single_epoch_spike():
    labels = ["N2", "N2", "W", "N2", "N2"]
    assert mode_filter_labels(labels, window=3) == ["N2", "N2", "N2", "N2", "N2"]


def test_viterbi_decode_uses_transition_counts_and_probability_shape():
    train_sequences = [["W", "N1", "N2", "N2", "R"], ["W", "N1", "N2", "N3", "N2"]]
    log_transitions = build_transition_log_probs(train_sequences, smoothing=1.0)
    probabilities = np.full((4, 5), 0.03, dtype=np.float32)
    probabilities[:, LABEL_TO_INDEX["N2"]] = 0.80
    probabilities[0, LABEL_TO_INDEX["W"]] = 0.90
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    decoded = viterbi_decode(probabilities, log_transitions)

    assert len(decoded) == 4
    assert decoded[0] == "W"
    assert set(decoded).issubset({"W", "N1", "N2", "N3", "R"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_models_smoothing.py::test_mode_filter_labels_removes_single_epoch_spike tests/test_models_smoothing.py::test_viterbi_decode_uses_transition_counts_and_probability_shape -v
```

Expected output:

```text
E   ModuleNotFoundError: No module named 'sleep_stage.smoothing'
```

- [ ] **Step 3: Implement smoothing utilities**

```python
# /Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/smoothing.py
from __future__ import annotations

from collections import Counter

import numpy as np

from sleep_stage.config import INDEX_TO_LABEL, LABELS, LABEL_TO_INDEX


def mode_filter_labels(labels: list[str], window: int = 3) -> list[str]:
    if window % 2 != 1:
        raise ValueError("window must be odd")
    radius = window // 2
    output: list[str] = []
    for index, label in enumerate(labels):
        left = max(0, index - radius)
        right = min(len(labels), index + radius + 1)
        counts = Counter(labels[left:right])
        output.append(counts.most_common(1)[0][0] if counts else label)
    return output


def build_transition_log_probs(train_sequences: list[list[str]], smoothing: float = 1.0) -> np.ndarray:
    counts = np.full((len(LABELS), len(LABELS)), smoothing, dtype=np.float64)
    for sequence in train_sequences:
        encoded = [LABEL_TO_INDEX[label] for label in sequence]
        for previous, current in zip(encoded[:-1], encoded[1:], strict=False):
            counts[previous, current] += 1.0
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    return np.log(probabilities)


def viterbi_decode(probabilities: np.ndarray, transition_log_probs: np.ndarray) -> list[str]:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(LABELS):
        raise ValueError(f"probabilities must have shape (n_epochs, {len(LABELS)})")
    emissions = np.log(np.clip(probabilities.astype(np.float64), 1e-12, 1.0))
    n_epochs, n_labels = emissions.shape
    scores = np.zeros((n_epochs, n_labels), dtype=np.float64)
    backpointers = np.zeros((n_epochs, n_labels), dtype=np.int64)
    scores[0] = emissions[0]
    for epoch in range(1, n_epochs):
        transition_scores = scores[epoch - 1][:, None] + transition_log_probs
        backpointers[epoch] = np.argmax(transition_scores, axis=0)
        scores[epoch] = emissions[epoch] + np.max(transition_scores, axis=0)
    labels = np.zeros(n_epochs, dtype=np.int64)
    labels[-1] = int(np.argmax(scores[-1]))
    for epoch in range(n_epochs - 2, -1, -1):
        labels[epoch] = backpointers[epoch + 1, labels[epoch + 1]]
    return [INDEX_TO_LABEL[int(index)] for index in labels]
```

- [ ] **Step 4: Run smoothing tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_models_smoothing.py -v
```

Expected output:

```text
tests/test_models_smoothing.py::test_mode_filter_labels_removes_single_epoch_spike PASSED
tests/test_models_smoothing.py::test_viterbi_decode_uses_transition_counts_and_probability_shape PASSED
```

- [ ] **Step 5: Commit**

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
git add src/sleep_stage/smoothing.py tests/test_models_smoothing.py
git commit -m "feat: smooth sleep stage probabilities"
```

Expected output contains:

```text
feat: smooth sleep stage probabilities
 2 files changed
```

## Task 7: Feature Cache And Experiment Pipeline

**Files:**
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/data.py`
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/pipeline.py`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_data.py`

- [ ] **Step 1: Write failing cache tests**

```python
# Append to /Users/temicide/Documents/5_domain_final/Sleep-Stage/tests/test_data.py
from sleep_stage.pipeline import write_submission_from_labels


def test_write_submission_from_labels_validates_and_preserves_sample_order(tmp_path: Path):
    sample = pd.DataFrame({"id": ["test001_00001", "test001_00000"], "labels": ["N2", "N2"]})
    sample_path = tmp_path / "sample_submission.csv"
    output_path = tmp_path / "submission.csv"
    sample.to_csv(sample_path, index=False)

    validation = write_submission_from_labels(
        labels_by_id={"test001_00000": "R", "test001_00001": "W"},
        sample_submission_path=sample_path,
        output_path=output_path,
    )

    written = pd.read_csv(output_path)
    assert written.to_dict("records") == [
        {"id": "test001_00001", "labels": "W"},
        {"id": "test001_00000", "labels": "R"},
    ]
    assert validation.rows == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_data.py::test_write_submission_from_labels_validates_and_preserves_sample_order -v
```

Expected output:

```text
E   ModuleNotFoundError: No module named 'sleep_stage.pipeline'
```

- [ ] **Step 3: Add pipeline orchestration functions**

```python
# /Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/pipeline.py
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sleep_stage.config import INDEX_TO_LABEL, LABELS, ProjectPaths, build_paths
from sleep_stage.context import add_context_features
from sleep_stage.data import (
    epoch_dataframe_from_recording,
    iter_train_recordings,
    list_test_segment_paths,
    load_train_recording,
    read_test_segments,
    validate_submission,
)
from sleep_stage.features import extract_epoch_features
from sleep_stage.models import fit_final_model, get_feature_columns, run_grouped_cv
from sleep_stage.smoothing import build_transition_log_probs, viterbi_decode


def build_train_feature_table(paths: ProjectPaths) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for recording_id, recording_path in iter_train_recordings(paths.train_dir):
        frame = load_train_recording(recording_path)
        epochs = epoch_dataframe_from_recording(frame, recording_id)
        for epoch in epochs.itertuples(index=False):
            epoch_frame = frame.iloc[int(epoch.start_row):int(epoch.end_row)]
            row = {
                "recording_id": recording_id,
                "epoch_index": int(epoch.epoch_index),
                "label": str(epoch.label),
            }
            row.update(extract_epoch_features(epoch_frame))
            rows.append(row)
    table = pd.DataFrame(rows)
    metadata = ["recording_id", "epoch_index", "label"]
    features = sorted(column for column in table.columns if column not in metadata)
    return table[metadata + features]


def build_test_feature_table(paths: ProjectPaths) -> pd.DataFrame:
    segment_paths = list_test_segment_paths(paths.test_dir)
    segments = read_test_segments(paths.sample_submission, segment_paths)
    rows: list[dict[str, object]] = []
    for segment_id, frame in segments.items():
        recording_id = segment_id.split("_", maxsplit=1)[0]
        epoch_index = int(segment_id.rsplit("_", maxsplit=1)[1])
        row = {
            "recording_id": recording_id,
            "segment_id": segment_id,
            "epoch_index": epoch_index,
        }
        row.update(extract_epoch_features(frame))
        rows.append(row)
    table = pd.DataFrame(rows)
    metadata = ["recording_id", "segment_id", "epoch_index"]
    features = sorted(column for column in table.columns if column not in metadata)
    return table[metadata + features]


def load_or_build_features(paths: ProjectPaths, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = paths.cache_dir / "train_epoch_features.parquet"
    test_cache = paths.cache_dir / "test_epoch_features.parquet"
    if not force and train_cache.exists() and test_cache.exists():
        return pd.read_parquet(train_cache), pd.read_parquet(test_cache)
    train_table = build_train_feature_table(paths)
    test_table = build_test_feature_table(paths)
    train_table.to_parquet(train_cache, index=False)
    test_table.to_parquet(test_cache, index=False)
    return train_table, test_table


def train_sequences_from_table(table: pd.DataFrame) -> list[list[str]]:
    sequences: list[list[str]] = []
    for _, group in table.sort_values(["recording_id", "epoch_index"]).groupby("recording_id", sort=False):
        sequences.append(group["label"].astype(str).tolist())
    return sequences


def predict_labels_by_segment(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    model_name: str,
    use_context: bool,
    use_viterbi: bool,
) -> dict[str, str]:
    train_features = add_context_features(train_table) if use_context else train_table
    test_features = add_context_features(test_table) if use_context else test_table
    model, feature_cols = fit_final_model(train_features, model_name=model_name)
    probabilities = model.predict_proba(test_features[feature_cols].astype(np.float32))
    aligned = np.zeros((len(test_features), len(LABELS)), dtype=np.float32)
    for local_index, class_index in enumerate(model.classes_):
        aligned[:, int(class_index)] = probabilities[:, local_index]
    aligned = aligned / np.clip(aligned.sum(axis=1, keepdims=True), 1e-9, None)
    labels: list[str] = []
    if use_viterbi:
        transition_log_probs = build_transition_log_probs(train_sequences_from_table(train_table))
        for _, group in test_features.groupby("recording_id", sort=False):
            group_probabilities = aligned[group.index.to_numpy()]
            labels.extend(viterbi_decode(group_probabilities, transition_log_probs))
    else:
        labels = [INDEX_TO_LABEL[int(index)] for index in aligned.argmax(axis=1)]
    return dict(zip(test_features["segment_id"].astype(str), labels, strict=True))


def write_submission_from_labels(
    labels_by_id: dict[str, str],
    sample_submission_path: Path,
    output_path: Path,
):
    sample = pd.read_csv(sample_submission_path)
    missing = sorted(set(sample["id"].astype(str)) - set(labels_by_id))
    if missing:
        raise ValueError(f"missing predictions for ids: {missing[:10]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": sample["id"].astype(str),
            "labels": [labels_by_id[segment_id] for segment_id in sample["id"].astype(str)],
        }
    )
    submission.to_csv(output_path, index=False)
    return validate_submission(output_path, sample_submission_path)


def run_experiments(paths: ProjectPaths | None = None, force_features: bool = False) -> pd.DataFrame:
    resolved = paths or build_paths()
    train_table, _ = load_or_build_features(resolved, force=force_features)
    experiments = [
        ("static_hgb", train_table, "hgb"),
        ("static_extra_trees", train_table, "extra_trees"),
        ("context_hgb", add_context_features(train_table), "hgb"),
        ("context_extra_trees", add_context_features(train_table), "extra_trees"),
    ]
    rows = []
    for name, table, model_name in experiments:
        report = run_grouped_cv(table, model_name=model_name)
        rows.append(
            {
                "experiment": name,
                "model_name": model_name,
                "n_features": len(get_feature_columns(table)),
                "weighted_f1_mean": report.weighted_f1_mean,
                "weighted_f1_std": report.weighted_f1_std,
                "fold_scores": report.fold_scores,
            }
        )
        joblib.dump(report, resolved.cache_dir / f"{name}_cv_report.joblib")
    results = pd.DataFrame(rows).sort_values("weighted_f1_mean", ascending=False)
    results.to_csv(resolved.cache_dir / "experiment_results.csv", index=False)
    return results


def generate_submission(paths: ProjectPaths | None = None) -> Path:
    resolved = paths or build_paths()
    train_table, test_table = load_or_build_features(resolved, force=False)
    labels_by_id = predict_labels_by_segment(
        train_table=train_table,
        test_table=test_table,
        model_name="hgb",
        use_context=True,
        use_viterbi=True,
    )
    output_path = resolved.output_dir / "submission.csv"
    validation = write_submission_from_labels(labels_by_id, resolved.sample_submission, output_path)
    print(f"Validated submission: rows={validation.rows}, labels={sorted(set(validation.valid_labels))}")
    return output_path
```

- [ ] **Step 4: Run pipeline tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest tests/test_data.py::test_write_submission_from_labels_validates_and_preserves_sample_order -v
```

Expected output:

```text
tests/test_data.py::test_write_submission_from_labels_validates_and_preserves_sample_order PASSED
```

- [ ] **Step 5: Run full unit test suite**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python -m pytest -v
```

Expected output:

```text
tests/test_data.py::test_constants_match_competition_spec PASSED
tests/test_data.py::test_find_competition_root_prefers_kaggle_then_local PASSED
tests/test_data.py::test_validate_submission_accepts_complete_known_labels PASSED
tests/test_data.py::test_epoch_dataframe_from_recording_returns_one_row_per_480_samples PASSED
tests/test_data.py::test_list_and_read_test_segments_follow_sample_submission_order PASSED
tests/test_data.py::test_write_submission_from_labels_validates_and_preserves_sample_order PASSED
tests/test_features_context.py::test_extract_epoch_features_has_deterministic_finite_values PASSED
tests/test_features_context.py::test_extract_feature_table_returns_metadata_and_sorted_feature_columns PASSED
tests/test_features_context.py::test_add_context_features_preserves_rows_and_adds_temporal_columns PASSED
tests/test_models_smoothing.py::test_build_model_returns_predict_proba_estimator PASSED
tests/test_models_smoothing.py::test_run_grouped_cv_returns_fold_metrics_and_oof_probabilities PASSED
tests/test_models_smoothing.py::test_mode_filter_labels_removes_single_epoch_spike PASSED
tests/test_models_smoothing.py::test_viterbi_decode_uses_transition_counts_and_probability_shape PASSED
```

- [ ] **Step 6: Commit**

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
git add src/sleep_stage/data.py src/sleep_stage/pipeline.py tests/test_data.py
git commit -m "feat: build experiment pipeline"
```

Expected output contains:

```text
feat: build experiment pipeline
 3 files changed
```

## Task 8: Local Full-Data Experiment Run

**Files:**
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/cache/experiment_results.csv`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/cache/static_hgb_cv_report.joblib`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/cache/static_extra_trees_cv_report.joblib`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/cache/context_hgb_cv_report.joblib`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/cache/context_extra_trees_cv_report.joblib`

- [ ] **Step 1: Build feature caches**

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
PY
```

Expected output:

```text
(66745, 178)
(7832, 178)
{'N2': 33786, 'W': 15828, 'N1': 7753, 'R': 7033, 'N3': 2345}
```

The feature count is 175 numeric features plus 3 metadata columns in both train and test tables.

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

Expected output:

```text
        experiment   model_name  n_features  weighted_f1_mean  weighted_f1_std
       context_hgb          hgb        2976
context_extra_trees  extra_trees        2976
        static_hgb          hgb         175
 static_extra_trees  extra_trees         175
```

If `context_hgb` scores below `0.50`, inspect `/Users/temicide/Documents/5_domain_final/Sleep-Stage/cache/context_hgb_cv_report.joblib` for per-class collapse before producing a leaderboard-candidate file.

- [ ] **Step 3: Print per-class F1 and confusion matrix for the best experiment**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python - <<'PY'
import joblib
from sleep_stage.config import LABELS

report = joblib.load("cache/context_hgb_cv_report.joblib")
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

The N3 and R rows must contain nonzero predicted support. If N3 is always predicted as N2, adjust class balance in `/Users/temicide/Documents/5_domain_final/Sleep-Stage/src/sleep_stage/models.py` by reducing HGB `learning_rate` to `0.035`, increasing `max_iter` to `650`, and rerun this task.

- [ ] **Step 4: Commit experiment metadata**

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
git add cache/experiment_results.csv
git commit -m "exp: record grouped sleep stage baselines"
```

Expected output contains:

```text
exp: record grouped sleep stage baselines
 1 file changed
```

## Task 9: Kaggle Notebook Wrapper

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/sleep_stage_solution.ipynb`

- [ ] **Step 1: Create the notebook with these cells**

Cell 1:

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path("/kaggle/working/Sleep-Stage")
LOCAL_ROOT = Path("/Users/temicide/Documents/5_domain_final/Sleep-Stage")
if LOCAL_ROOT.exists():
    PROJECT_ROOT = LOCAL_ROOT

src_path = PROJECT_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print(f"Using project root: {PROJECT_ROOT}")
```

Cell 2:

```python
from sleep_stage.config import build_paths
from sleep_stage.pipeline import load_or_build_features, run_experiments

paths = build_paths(project_root=PROJECT_ROOT)
print(paths)
train_table, test_table = load_or_build_features(paths, force=False)
print("train_table", train_table.shape)
print("test_table", test_table.shape)
print(train_table["label"].value_counts().to_dict())
```

Cell 3:

```python
results = run_experiments(paths=paths, force_features=False)
print(results[["experiment", "model_name", "n_features", "weighted_f1_mean", "weighted_f1_std"]].to_string(index=False))
best = results.iloc[0].to_dict()
print("Best grouped-CV experiment:", best)
```

Cell 4:

```python
from sleep_stage.pipeline import generate_submission

submission_path = generate_submission(paths)
print(submission_path)
```

Cell 5:

```python
import pandas as pd
from sleep_stage.data import validate_submission

validation = validate_submission(paths.output_dir / "submission.csv", paths.sample_submission)
submission = pd.read_csv(paths.output_dir / "submission.csv")
print(validation)
print(submission.head().to_string(index=False))
print(submission["labels"].value_counts().to_string())
assert len(submission) == 7832
```

- [ ] **Step 2: Generate the `.ipynb` file using nbformat**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python - <<'PY'
from pathlib import Path
import nbformat as nbf

cells = [
    '''from pathlib import Path
import sys

PROJECT_ROOT = Path("/kaggle/working/Sleep-Stage")
LOCAL_ROOT = Path("/Users/temicide/Documents/5_domain_final/Sleep-Stage")
if LOCAL_ROOT.exists():
    PROJECT_ROOT = LOCAL_ROOT

src_path = PROJECT_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print(f"Using project root: {PROJECT_ROOT}")''',
    '''from sleep_stage.config import build_paths
from sleep_stage.pipeline import load_or_build_features, run_experiments

paths = build_paths(project_root=PROJECT_ROOT)
print(paths)
train_table, test_table = load_or_build_features(paths, force=False)
print("train_table", train_table.shape)
print("test_table", test_table.shape)
print(train_table["label"].value_counts().to_dict())''',
    '''results = run_experiments(paths=paths, force_features=False)
print(results[["experiment", "model_name", "n_features", "weighted_f1_mean", "weighted_f1_std"]].to_string(index=False))
best = results.iloc[0].to_dict()
print("Best grouped-CV experiment:", best)''',
    '''from sleep_stage.pipeline import generate_submission

submission_path = generate_submission(paths)
print(submission_path)''',
    '''import pandas as pd
from sleep_stage.data import validate_submission

validation = validate_submission(paths.output_dir / "submission.csv", paths.sample_submission)
submission = pd.read_csv(paths.output_dir / "submission.csv")
print(validation)
print(submission.head().to_string(index=False))
print(submission["labels"].value_counts().to_string())
assert len(submission) == 7832''',
]
notebook = nbf.v4.new_notebook()
notebook["cells"] = [nbf.v4.new_code_cell(cell) for cell in cells]
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

- [ ] **Step 3: Verify the notebook does not submit through Kaggle API or CLI**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python - <<'PY'
from pathlib import Path

text = Path("sleep_stage_solution.ipynb").read_text()
blocked = ["kaggle competitions submit", "KaggleApi", ".submit("]
found = [token for token in blocked if token in text]
print("blocked tokens found:", found)
assert not found
PY
```

Expected output:

```text
blocked tokens found: []
```

- [ ] **Step 4: Commit**

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
git add sleep_stage_solution.ipynb
git commit -m "feat: add kaggle notebook solution"
```

Expected output contains:

```text
feat: add kaggle notebook solution
 1 file changed
```

## Task 10: Submission Generation And Validation

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/working/submission.csv`
- Modify: `/Users/temicide/Documents/5_domain_final/Sleep-Stage/submission_log.md`

- [ ] **Step 1: Generate local submission**

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
validation = validate_submission(paths.output_dir / "submission.csv", paths.sample_submission)
submission = pd.read_csv(paths.output_dir / "submission.csv")
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

- [ ] **Step 3: Record the submission attempt**

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

Do not add a Kaggle API or CLI submission command to any file.

- [ ] **Step 4: Commit submission log, not generated CSV**

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
git add submission_log.md
git commit -m "docs: add sleep stage submission log"
```

Expected output contains:

```text
docs: add sleep stage submission log
 1 file changed
```

## Task 11: Self-Review And Acceptance Checks

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

- [ ] **Step 3: Verify no automated submission commands exist**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
rg -n "kaggle competitions submit|KaggleApi|\\.submit\\(" .
```

Expected output:

```text
```

- [ ] **Step 4: Verify final submission file**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
python - <<'PY'
from sleep_stage.config import build_paths
from sleep_stage.data import validate_submission

paths = build_paths()
validation = validate_submission(paths.output_dir / "submission.csv", paths.sample_submission)
print(validation.rows)
print(validation.id_match)
print(sorted(set(validation.valid_labels)))
PY
```

Expected output:

```text
7832
True
['N1', 'N2', 'N3', 'R', 'W']
```

- [ ] **Step 5: Review spec coverage**

Confirm the following mapping before declaring completion:

```text
Kaggle notebook end-to-end: Task 9
/kaggle/input read path with local fallback: Task 1 and Task 9
/kaggle/working/submission.csv generation: Task 7 and Task 10
CSV validation before completion: Task 1, Task 7, Task 10, Task 11
No automated Kaggle submission: Task 9 and Task 11
480-row epoch parsing: Task 2
Grouped validation by recording: Task 5 and Task 8
Weighted F1 and per-class F1 reporting: Task 5 and Task 8
Rich epoch features: Task 3
Neighbor and rolling context: Task 4
Temporal smoothing and Viterbi decoding: Task 6
Submission strategy and manual log: Task 10
```

- [ ] **Step 6: Final commit after acceptance checks**

```bash
cd /Users/temicide/Documents/5_domain_final/Sleep-Stage
git status --short
git add pyproject.toml src tests sleep_stage_solution.ipynb submission_log.md cache/experiment_results.csv
git commit -m "feat: complete sleep stage kaggle pipeline"
```

Expected output contains:

```text
feat: complete sleep stage kaggle pipeline
 files changed
```

## Self-Review Notes

**Spec coverage:** This plan covers Kaggle Notebook execution, Kaggle-first input discovery with local fallbacks, `/kaggle/working/submission.csv` generation, submission validation, no automated Kaggle submission, 480-row epoch parsing, grouped validation by full recording, weighted and per-class F1 reporting, rich epoch features, neighbor and rolling context features, Viterbi smoothing, and a manual submission log.

**Placeholder scan:** The plan avoids unresolved implementation placeholders. Code-editing steps include concrete file paths, code blocks, commands, and expected outputs; future leaderboard log updates require measured numeric values before a row is appended.

**Type consistency:** Shared names are consistent across tasks: `EPOCH_ROWS`, `SIGNAL_COLUMNS`, `LABELS`, `CompetitionPaths`, `SubmissionValidation`, `EpochTable`, `FeatureBundle`, `FoldResult`, `ExperimentResult`, `build_context_features`, `train_transition_matrix`, `viterbi_decode`, and `run_submission`.

## Execution Handoff

Plan complete and saved to `/Users/temicide/Documents/5_domain_final/Sleep-Stage/plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
