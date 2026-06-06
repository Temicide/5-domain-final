# Chest Disease Colab A100 Solution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Google Colab-ready A100 notebook and reusable local pipeline for multi-label chest X-ray disease detection that downloads Kaggle data securely, trains validated models, tunes thresholds from OOF predictions, and writes `/content/submission.csv` without automated Kaggle submission.

**Architecture:** Keep reusable code in a small Python package under `Chest-Disease/src/chest_disease`, then generate a self-contained Colab notebook that writes those package files into `/content/Chest-Disease/src` before importing them. The pipeline separates Kaggle/data bootstrap, schema validation, folds/metrics/thresholds, image datasets/transforms, model factories, training/inference, and final submission writing so each unit is testable locally before GPU training.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, timm, torchxrayvision, pandas, numpy, scikit-learn, iterative-stratification, Pillow, pytest, nbformat, Kaggle CLI/API, Google Colab A100.

---

## Scope Check

The spec describes one competition workflow: a Colab notebook that trains a multi-label chest X-ray classifier and writes a validated Kaggle CSV. The workflow includes several components, but they are tightly coupled and should be implemented as one plan because every component is needed for the same testable artifact: `/content/submission.csv`.

External CXR-pretrained weights are useful but must be gated by a runtime setting because the competition rules may disallow external-data-derived weights. The implementation must support both `allow_external_weights=True` for TorchXRayVision and `allow_external_weights=False` for ImageNet/timm fallback models.

## File Structure

- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/pyproject.toml`: local package metadata, pytest configuration, and dependency declarations.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/__init__.py`: package marker and version string.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/config.py`: constants for labels, paths, competition slug, model settings, and typed path dataclasses.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/kaggle_io.py`: secure Kaggle credential setup, download command construction, archive extraction, and path discovery.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/data.py`: CSV loading, image path resolution, train/test schema validation, and submission validation.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/folds.py`: multi-label stratified fold creation with deterministic fallback.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/metrics.py`: ROC-AUC, average precision, F1, threshold tuning, and `No Finding` consistency rules.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/dataset.py`: PyTorch `Dataset`, image loading, resize-pad transform, center-crop transform, and train/eval augmentation builders.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/models.py`: TorchXRayVision and timm model factories with 13-output heads.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/train.py`: A100 mixed-precision training loop, validation inference, checkpoint saving, and test inference.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/pipeline.py`: end-to-end orchestration for data setup, fold training, OOF/test prediction writing, threshold tuning, and submission generation.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/scripts/build_colab_notebook.py`: deterministic notebook builder.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/notebooks/chest_disease_colab_a100_solution.ipynb`: generated self-contained Colab notebook that embeds the local package source and does not require cloning this repository in Colab.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_config_kaggle_io.py`: tests for constants, paths, credentials, download commands, and extraction.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_data_submission.py`: tests for CSV schema, image path resolution, and submission validation.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_folds_metrics.py`: tests for fold creation, metric outputs, threshold tuning, and `No Finding` post-processing.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_dataset_models.py`: tests for image transforms and model output shapes.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_pipeline_smoke.py`: tiny end-to-end smoke test that uses a fake trainer to produce a valid submission.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/outputs/oof/.gitkeep`: local OOF artifact directory marker.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/outputs/test_preds/.gitkeep`: local test prediction artifact directory marker.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/outputs/thresholds/.gitkeep`: local threshold artifact directory marker.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/outputs/submissions/.gitkeep`: local submission artifact directory marker.
- Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/submission_log.md`: manual record of CV metrics, public LB feedback, and selected final candidates.

## Known Data Contract

Use these exact constants:

```python
COMPETITION_SLUG = "chest-disease-detection"
LOCAL_COMPETITION_DIRNAME = "individual-test-chest-disease-detection"
ID_COLUMN = "filename"
LABEL_COLUMNS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]
DISEASE_LABEL_COLUMNS = [label for label in LABEL_COLUMNS if label != "No Finding"]
EXPECTED_COLUMNS = [ID_COLUMN] + LABEL_COLUMNS
```

Use these exact path defaults:

```python
PROJECT_ROOT = Path("/Users/temicide/Documents/5_domain_final/Chest-Disease")
COLAB_ROOT = Path("/content")
COLAB_INPUT_ROOT = COLAB_ROOT / "input"
COLAB_WORKING_DIR = COLAB_ROOT / "working"
COLAB_COMPETITION_DIR = COLAB_INPUT_ROOT / COMPETITION_SLUG
COLAB_ALT_COMPETITION_DIR = COLAB_INPUT_ROOT / LOCAL_COMPETITION_DIRNAME
COLAB_SUBMISSION_PATH = COLAB_ROOT / "submission.csv"
COLAB_PACKAGE_ROOT = COLAB_ROOT / "Chest-Disease"
COLAB_PACKAGE_SRC = COLAB_PACKAGE_ROOT / "src"
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / LOCAL_COMPETITION_DIRNAME
LOCAL_IMAGE_DIR = LOCAL_DATA_DIR / "images" / "images"
LOCAL_OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOCAL_SUBMISSION_PATH = LOCAL_OUTPUT_DIR / "submissions" / "submission.csv"
```

## Implementation Tasks

### Task 1: Package Skeleton And Constants

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/pyproject.toml`
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/__init__.py`
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/config.py`
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_config_kaggle_io.py`
- Create: output `.gitkeep` files under `/Users/temicide/Documents/5_domain_final/Chest-Disease/outputs/`

- [ ] **Step 1: Create directories**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
mkdir -p src/chest_disease scripts notebooks tests outputs/oof outputs/test_preds outputs/thresholds outputs/submissions
touch outputs/oof/.gitkeep outputs/test_preds/.gitkeep outputs/thresholds/.gitkeep outputs/submissions/.gitkeep
```

Expected: command exits with status `0` and prints nothing.

- [ ] **Step 2: Write the failing constants test**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_config_kaggle_io.py`:

```python
from pathlib import Path

from chest_disease import __version__
from chest_disease.config import (
    COLAB_ALT_COMPETITION_DIR,
    COLAB_COMPETITION_DIR,
    COLAB_INPUT_ROOT,
    COLAB_PACKAGE_ROOT,
    COLAB_PACKAGE_SRC,
    COLAB_SUBMISSION_PATH,
    COLAB_WORKING_DIR,
    COMPETITION_SLUG,
    DISEASE_LABEL_COLUMNS,
    EXPECTED_COLUMNS,
    ID_COLUMN,
    LABEL_COLUMNS,
    LOCAL_COMPETITION_DIRNAME,
    LOCAL_DATA_DIR,
    LOCAL_IMAGE_DIR,
    LOCAL_SUBMISSION_PATH,
    PROJECT_ROOT,
    RunConfig,
)


def test_constants_match_chest_disease_spec():
    assert __version__ == "0.1.0"
    assert COMPETITION_SLUG == "chest-disease-detection"
    assert LOCAL_COMPETITION_DIRNAME == "individual-test-chest-disease-detection"
    assert ID_COLUMN == "filename"
    assert LABEL_COLUMNS == [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
    ]
    assert DISEASE_LABEL_COLUMNS == [label for label in LABEL_COLUMNS if label != "No Finding"]
    assert EXPECTED_COLUMNS == [ID_COLUMN] + LABEL_COLUMNS
    assert PROJECT_ROOT == Path("/Users/temicide/Documents/5_domain_final/Chest-Disease")
    assert COLAB_INPUT_ROOT == Path("/content/input")
    assert COLAB_WORKING_DIR == Path("/content/working")
    assert COLAB_COMPETITION_DIR == Path("/content/input/chest-disease-detection")
    assert COLAB_ALT_COMPETITION_DIR == Path("/content/input/individual-test-chest-disease-detection")
    assert COLAB_SUBMISSION_PATH == Path("/content/submission.csv")
    assert COLAB_PACKAGE_ROOT == Path("/content/Chest-Disease")
    assert COLAB_PACKAGE_SRC == Path("/content/Chest-Disease/src")
    assert LOCAL_DATA_DIR == PROJECT_ROOT / "data" / LOCAL_COMPETITION_DIRNAME
    assert LOCAL_IMAGE_DIR == LOCAL_DATA_DIR / "images" / "images"
    assert LOCAL_SUBMISSION_PATH == PROJECT_ROOT / "outputs" / "submissions" / "submission.csv"


def test_run_config_defaults_are_a100_friendly():
    config = RunConfig()
    assert config.image_size == 512
    assert config.batch_size == 32
    assert config.num_folds == 5
    assert config.seed == 42
    assert config.allow_external_weights is True
    assert config.use_amp is True
    assert config.output_mode == "binary"
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_config_kaggle_io.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'chest_disease'
```

- [ ] **Step 4: Write package metadata and constants**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "chest-disease"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.23",
  "pandas>=1.5",
  "pillow>=9.0",
  "scikit-learn>=1.2",
  "torch>=2.0",
  "torchvision>=0.15",
  "timm>=0.9",
  "iterative-stratification>=0.1.7",
  "nbformat>=5.9",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-q"
```

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

COMPETITION_SLUG = "chest-disease-detection"
LOCAL_COMPETITION_DIRNAME = "individual-test-chest-disease-detection"
ID_COLUMN = "filename"
LABEL_COLUMNS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]
DISEASE_LABEL_COLUMNS = [label for label in LABEL_COLUMNS if label != "No Finding"]
EXPECTED_COLUMNS = [ID_COLUMN] + LABEL_COLUMNS

PROJECT_ROOT = Path("/Users/temicide/Documents/5_domain_final/Chest-Disease")
COLAB_ROOT = Path("/content")
COLAB_INPUT_ROOT = COLAB_ROOT / "input"
COLAB_WORKING_DIR = COLAB_ROOT / "working"
COLAB_COMPETITION_DIR = COLAB_INPUT_ROOT / COMPETITION_SLUG
COLAB_ALT_COMPETITION_DIR = COLAB_INPUT_ROOT / LOCAL_COMPETITION_DIRNAME
COLAB_SUBMISSION_PATH = COLAB_ROOT / "submission.csv"
COLAB_PACKAGE_ROOT = COLAB_ROOT / "Chest-Disease"
COLAB_PACKAGE_SRC = COLAB_PACKAGE_ROOT / "src"
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / LOCAL_COMPETITION_DIRNAME
LOCAL_IMAGE_DIR = LOCAL_DATA_DIR / "images" / "images"
LOCAL_OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOCAL_SUBMISSION_PATH = LOCAL_OUTPUT_DIR / "submissions" / "submission.csv"


@dataclass(frozen=True)
class ProjectPaths:
    competition_dir: Path
    image_dir: Path
    train_csv: Path
    test_submission_csv: Path
    working_dir: Path
    submission_path: Path


@dataclass(frozen=True)
class RunConfig:
    image_size: int = 512
    batch_size: int = 32
    num_folds: int = 5
    seed: int = 42
    allow_external_weights: bool = True
    use_amp: bool = True
    output_mode: str = "binary"
    model_name: str = "torchxrayvision_densenet121_all"
    epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
```

- [ ] **Step 5: Run constants tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_config_kaggle_io.py::test_constants_match_chest_disease_spec tests/test_config_kaggle_io.py::test_run_config_defaults_are_a100_friendly -v
```

Expected:

```text
PASSED
```

- [ ] **Step 6: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final
git add Chest-Disease/pyproject.toml Chest-Disease/src/chest_disease/__init__.py Chest-Disease/src/chest_disease/config.py Chest-Disease/tests/test_config_kaggle_io.py Chest-Disease/outputs
git commit -m "feat(chest): add package scaffold and constants"
```

Expected: commit succeeds with the listed files.

### Task 2: Kaggle Bootstrap And Path Resolution

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/kaggle_io.py`
- Modify: `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_config_kaggle_io.py`

- [ ] **Step 1: Add failing Kaggle IO tests**

Append to `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_config_kaggle_io.py`:

```python
import os
import zipfile

import pytest

from chest_disease.kaggle_io import (
    build_download_command,
    configure_kaggle_credentials,
    extract_archive,
    resolve_project_paths,
)


def test_build_download_command_uses_competition_slug():
    command = build_download_command(input_root=Path("/content/input"))
    assert command == [
        "kaggle",
        "competitions",
        "download",
        "-c",
        "chest-disease-detection",
        "-p",
        "/content/input",
    ]


def test_configure_kaggle_credentials_uses_env_without_printing(capsys):
    env = {"KAGGLE_USERNAME": "user-name", "KAGGLE_KEY": "secret-key"}
    status = configure_kaggle_credentials(env=env, kaggle_json_bytes=None, kaggle_dir=Path("/tmp/not-used"))
    captured = capsys.readouterr()
    assert status == "environment_variables"
    assert "secret-key" not in captured.out
    assert "user-name" not in captured.out


def test_configure_kaggle_credentials_writes_uploaded_json_securely(tmp_path: Path, capsys):
    kaggle_dir = tmp_path / ".kaggle"
    payload = b'{"username":"uploaded-user","key":"uploaded-secret"}'
    status = configure_kaggle_credentials(env={}, kaggle_json_bytes=payload, kaggle_dir=kaggle_dir)
    target = kaggle_dir / "kaggle.json"
    captured = capsys.readouterr()
    assert status == "uploaded_kaggle_json"
    assert target.read_bytes() == payload
    assert oct(target.stat().st_mode & 0o777) == "0o600"
    assert "uploaded-secret" not in captured.out


def test_configure_kaggle_credentials_raises_when_missing(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Kaggle credentials are missing"):
        configure_kaggle_credentials(env={}, kaggle_json_bytes=None, kaggle_dir=tmp_path)


def test_extract_archive_and_resolve_paths_prefers_colab(tmp_path: Path):
    input_root = tmp_path / "input"
    archive = input_root / "chest-disease-detection.zip"
    image_file = "chest-disease-detection/images/images/cxr00001.jpg"
    input_root.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("chest-disease-detection/train.csv", "filename,Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,Pneumothorax\n")
        zf.writestr("chest-disease-detection/test_submission.csv", "filename,Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,Pneumothorax\n")
        zf.writestr(image_file, b"fake")
    extracted = extract_archive(archive, input_root / "chest-disease-detection")
    paths = resolve_project_paths(
        colab_input_root=input_root,
        local_data_dir=tmp_path / "local-missing",
        working_dir=tmp_path / "working",
        submission_path=tmp_path / "submission.csv",
    )
    assert extracted == input_root / "chest-disease-detection"
    assert paths.competition_dir == input_root / "chest-disease-detection"
    assert paths.train_csv.name == "train.csv"
    assert paths.test_submission_csv.name == "test_submission.csv"
    assert paths.image_dir.name == "images"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_config_kaggle_io.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'chest_disease.kaggle_io'
```

- [ ] **Step 3: Implement Kaggle IO**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/kaggle_io.py`:

```python
from __future__ import annotations

import os
import zipfile
from collections.abc import MutableMapping
from pathlib import Path

from .config import (
    COLAB_COMPETITION_DIR,
    COLAB_INPUT_ROOT,
    COLAB_SUBMISSION_PATH,
    COLAB_WORKING_DIR,
    COMPETITION_SLUG,
    LOCAL_DATA_DIR,
    PROJECT_ROOT,
    ProjectPaths,
)


def build_download_command(input_root: Path = COLAB_INPUT_ROOT) -> list[str]:
    return ["kaggle", "competitions", "download", "-c", COMPETITION_SLUG, "-p", str(input_root)]


def configure_kaggle_credentials(
    env: MutableMapping[str, str] | None = None,
    kaggle_json_bytes: bytes | None = None,
    kaggle_dir: Path = Path("/root/.kaggle"),
) -> str:
    active_env = env if env is not None else os.environ
    if active_env.get("KAGGLE_USERNAME") and active_env.get("KAGGLE_KEY"):
        return "environment_variables"
    if kaggle_json_bytes is not None:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        target = kaggle_dir / "kaggle.json"
        target.write_bytes(kaggle_json_bytes)
        target.chmod(0o600)
        return "uploaded_kaggle_json"
    existing = kaggle_dir / "kaggle.json"
    if existing.exists():
        existing.chmod(0o600)
        return "existing_kaggle_json"
    raise RuntimeError(
        "Kaggle credentials are missing. Provide KAGGLE_USERNAME/KAGGLE_KEY, an existing kaggle.json, or upload kaggle.json in Colab."
    )


def extract_archive(archive_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(destination_dir.parent)
    if _valid_competition_dir(destination_dir):
        return destination_dir
    if _valid_competition_dir(destination_dir.parent):
        return destination_dir.parent
    raise FileNotFoundError(
        f"Expected extracted competition files under {destination_dir} or {destination_dir.parent}"
    )


def _valid_competition_dir(path: Path) -> bool:
    return (
        (path / "train.csv").exists()
        and (path / "test_submission.csv").exists()
        and (path / "images" / "images").exists()
    )


def resolve_project_paths(
    colab_input_root: Path = COLAB_INPUT_ROOT,
    local_data_dir: Path = LOCAL_DATA_DIR,
    working_dir: Path = COLAB_WORKING_DIR,
    submission_path: Path = COLAB_SUBMISSION_PATH,
) -> ProjectPaths:
    candidates = [
        colab_input_root / COMPETITION_SLUG,
        colab_input_root / "individual-test-chest-disease-detection",
        colab_input_root,
        local_data_dir,
    ]
    for competition_dir in candidates:
        if _valid_competition_dir(competition_dir):
            return ProjectPaths(
                competition_dir=competition_dir,
                image_dir=competition_dir / "images" / "images",
                train_csv=competition_dir / "train.csv",
                test_submission_csv=competition_dir / "test_submission.csv",
                working_dir=working_dir,
                submission_path=submission_path,
            )
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Competition files not found. Checked: {checked}")
```

- [ ] **Step 4: Run Kaggle IO tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_config_kaggle_io.py -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final
git add Chest-Disease/src/chest_disease/kaggle_io.py Chest-Disease/tests/test_config_kaggle_io.py
git commit -m "feat(chest): add kaggle bootstrap and path resolution"
```

Expected: commit succeeds.

### Task 3: Data Loading And Submission Validation

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/data.py`
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_data_submission.py`

- [ ] **Step 1: Write failing data tests**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_data_submission.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from chest_disease.config import EXPECTED_COLUMNS, LABEL_COLUMNS
from chest_disease.data import (
    SubmissionValidation,
    load_competition_frames,
    resolve_image_paths,
    validate_submission,
)


def write_csvs(root: Path) -> None:
    root.mkdir(parents=True)
    images = root / "images" / "images"
    images.mkdir(parents=True)
    train = pd.DataFrame(
        [
            ["cxr00004.jpg", 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            ["cxr00005.jpg", 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        columns=EXPECTED_COLUMNS,
    )
    test = pd.DataFrame(
        [
            ["cxr00001.jpg"] + [""] * len(LABEL_COLUMNS),
            ["cxr00002.jpg"] + [""] * len(LABEL_COLUMNS),
        ],
        columns=EXPECTED_COLUMNS,
    )
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test_submission.csv", index=False)
    for filename in ["cxr00001.jpg", "cxr00002.jpg", "cxr00004.jpg", "cxr00005.jpg"]:
        (images / filename).write_bytes(b"fake")


def test_load_competition_frames_preserves_schema(tmp_path: Path):
    write_csvs(tmp_path)
    train, test = load_competition_frames(tmp_path / "train.csv", tmp_path / "test_submission.csv")
    assert list(train.columns) == EXPECTED_COLUMNS
    assert list(test.columns) == EXPECTED_COLUMNS
    assert train[LABEL_COLUMNS].shape == (2, 13)
    assert test["filename"].tolist() == ["cxr00001.jpg", "cxr00002.jpg"]


def test_resolve_image_paths_requires_every_file(tmp_path: Path):
    write_csvs(tmp_path)
    train, _ = load_competition_frames(tmp_path / "train.csv", tmp_path / "test_submission.csv")
    resolved = resolve_image_paths(train, tmp_path / "images" / "images")
    assert resolved["image_path"].map(Path).map(lambda p: p.exists()).all()
    (tmp_path / "images" / "images" / "cxr00004.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="cxr00004.jpg"):
        resolve_image_paths(train, tmp_path / "images" / "images")


def test_validate_submission_accepts_binary_and_probability_outputs(tmp_path: Path):
    write_csvs(tmp_path)
    template = pd.read_csv(tmp_path / "test_submission.csv")
    binary = template.copy()
    for label in LABEL_COLUMNS:
        binary[label] = 0
    binary.loc[0, "No Finding"] = 1
    binary.loc[1, "Atelectasis"] = 1
    path = tmp_path / "submission.csv"
    binary.to_csv(path, index=False)
    result = validate_submission(path, tmp_path / "test_submission.csv", output_mode="binary")
    assert isinstance(result, SubmissionValidation)
    assert result.rows == 2
    assert result.columns_match is True
    assert result.filenames_match is True
    assert result.no_missing_labels is True
    assert result.values_in_range is True


def test_validate_submission_rejects_missing_label_values(tmp_path: Path):
    write_csvs(tmp_path)
    bad = pd.read_csv(tmp_path / "test_submission.csv")
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="missing label values"):
        validate_submission(tmp_path / "bad.csv", tmp_path / "test_submission.csv", output_mode="binary")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_data_submission.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'chest_disease.data'
```

- [ ] **Step 3: Implement data utilities**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/data.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import EXPECTED_COLUMNS, ID_COLUMN, LABEL_COLUMNS


@dataclass(frozen=True)
class SubmissionValidation:
    rows: int
    columns_match: bool
    filenames_match: bool
    no_missing_labels: bool
    values_in_range: bool


def _assert_columns(df: pd.DataFrame, source: Path) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{source} columns do not match expected competition schema")


def load_competition_frames(train_csv: Path, test_submission_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_submission_csv)
    _assert_columns(train, train_csv)
    _assert_columns(test, test_submission_csv)
    train[LABEL_COLUMNS] = train[LABEL_COLUMNS].astype(float)
    return train, test


def resolve_image_paths(frame: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    resolved = frame.copy()
    paths = []
    missing = []
    for filename in resolved[ID_COLUMN].astype(str):
        path = image_dir / filename
        paths.append(str(path))
        if not path.exists():
            missing.append(filename)
    if missing:
        raise FileNotFoundError("Missing image files: " + ", ".join(missing[:10]))
    resolved["image_path"] = paths
    return resolved


def validate_submission(submission_path: Path, template_path: Path, output_mode: str) -> SubmissionValidation:
    submission = pd.read_csv(submission_path)
    template = pd.read_csv(template_path)
    columns_match = list(submission.columns) == list(template.columns) == EXPECTED_COLUMNS
    if not columns_match:
        raise ValueError("submission columns do not match test_submission.csv")
    filenames_match = submission[ID_COLUMN].astype(str).tolist() == template[ID_COLUMN].astype(str).tolist()
    if not filenames_match:
        raise ValueError("submission filenames do not match test_submission.csv order")
    labels = submission[LABEL_COLUMNS]
    no_missing = not labels.isna().any().any()
    if not no_missing:
        raise ValueError("submission has missing label values")
    numeric = labels.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("submission has non-numeric label values")
    if output_mode == "binary":
        values_ok = numeric.isin([0, 1]).all().all()
    elif output_mode == "probability":
        values_ok = ((numeric >= 0.0) & (numeric <= 1.0)).all().all()
    else:
        raise ValueError("output_mode must be 'binary' or 'probability'")
    if not values_ok:
        raise ValueError(f"submission label values are invalid for {output_mode} mode")
    return SubmissionValidation(
        rows=len(submission),
        columns_match=columns_match,
        filenames_match=filenames_match,
        no_missing_labels=no_missing,
        values_in_range=values_ok,
    )
```

- [ ] **Step 4: Run data tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_data_submission.py -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final
git add Chest-Disease/src/chest_disease/data.py Chest-Disease/tests/test_data_submission.py
git commit -m "feat(chest): add data and submission validation"
```

Expected: commit succeeds.

### Task 4: Folds, Metrics, Thresholds, And No Finding Rules

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/folds.py`
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/metrics.py`
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_folds_metrics.py`

- [ ] **Step 1: Write failing fold and metric tests**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_folds_metrics.py`:

```python
import numpy as np
import pandas as pd

from chest_disease.config import DISEASE_LABEL_COLUMNS, EXPECTED_COLUMNS, LABEL_COLUMNS
from chest_disease.folds import make_multilabel_folds
from chest_disease.metrics import apply_no_finding_consistency, compute_metrics, tune_thresholds


def make_frame(n: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n):
        labels = [0] * len(LABEL_COLUMNS)
        if i % 3 == 0:
            labels[LABEL_COLUMNS.index("No Finding")] = 1
        else:
            labels[i % len(DISEASE_LABEL_COLUMNS)] = 1
        rows.append([f"cxr{i:05d}.jpg"] + labels)
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_make_multilabel_folds_is_deterministic_and_complete():
    frame = make_frame(30)
    folded = make_multilabel_folds(frame, num_folds=3, seed=7)
    assert sorted(folded["fold"].unique().tolist()) == [0, 1, 2]
    assert folded["fold"].notna().all()
    assert folded["filename"].tolist() == frame["filename"].tolist()
    again = make_multilabel_folds(frame, num_folds=3, seed=7)
    assert folded["fold"].tolist() == again["fold"].tolist()


def test_tune_thresholds_returns_one_threshold_per_label():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    thresholds = tune_thresholds(y_true, y_prob, labels=["A", "B"])
    assert set(thresholds) == {"A", "B"}
    assert all(0.05 <= value <= 0.95 for value in thresholds.values())


def test_apply_no_finding_consistency_makes_normal_exclusive():
    predictions = pd.DataFrame(
        {
            "Atelectasis": [1, 0],
            "Cardiomegaly": [0, 0],
            "Consolidation": [0, 0],
            "Edema": [0, 0],
            "Enlarged Cardiomediastinum": [0, 0],
            "Fracture": [0, 0],
            "Lung Lesion": [0, 0],
            "Lung Opacity": [0, 0],
            "No Finding": [1, 0],
            "Pleural Effusion": [0, 0],
            "Pleural Other": [0, 0],
            "Pneumonia": [0, 0],
            "Pneumothorax": [0, 0],
        }
    )
    fixed = apply_no_finding_consistency(predictions)
    assert fixed.loc[0, "No Finding"] == 0
    assert fixed.loc[1, "No Finding"] == 1


def test_compute_metrics_reports_macro_f1_and_per_label_f1():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    report = compute_metrics(y_true, y_prob, labels=["A", "B"], thresholds={"A": 0.5, "B": 0.5})
    assert report["macro_f1"] == 1.0
    assert report["per_label"]["A"]["f1"] == 1.0
    assert report["per_label"]["B"]["f1"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_folds_metrics.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'chest_disease.folds'
```

- [ ] **Step 3: Implement folds**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/folds.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import LABEL_COLUMNS


def make_multilabel_folds(frame: pd.DataFrame, num_folds: int, seed: int) -> pd.DataFrame:
    result = frame.copy()
    y = result[LABEL_COLUMNS].astype(int).to_numpy()
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        splitter = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits = splitter.split(np.zeros(len(result)), y)
    except Exception:
        splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits = splitter.split(result)
    result["fold"] = -1
    for fold, (_, valid_idx) in enumerate(splits):
        result.loc[result.index[valid_idx], "fold"] = fold
    if (result["fold"] < 0).any():
        raise RuntimeError("fold assignment failed")
    return result
```

- [ ] **Step 4: Implement metrics**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/metrics.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .config import DISEASE_LABEL_COLUMNS, LABEL_COLUMNS


def tune_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    grid: np.ndarray | None = None,
) -> dict[str, float]:
    thresholds = grid if grid is not None else np.linspace(0.05, 0.95, 91)
    best: dict[str, float] = {}
    for index, label in enumerate(labels):
        scores = []
        for threshold in thresholds:
            pred = (y_prob[:, index] >= threshold).astype(int)
            scores.append(f1_score(y_true[:, index], pred, zero_division=0))
        best[label] = float(thresholds[int(np.argmax(scores))])
    return best


def apply_no_finding_consistency(predictions: pd.DataFrame) -> pd.DataFrame:
    fixed = predictions.copy()
    disease_positive = fixed[DISEASE_LABEL_COLUMNS].sum(axis=1) > 0
    fixed.loc[disease_positive, "No Finding"] = 0
    fixed.loc[~disease_positive, "No Finding"] = 1
    return fixed[LABEL_COLUMNS]


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    thresholds: dict[str, float],
) -> dict[str, object]:
    pred = np.zeros_like(y_prob, dtype=int)
    for index, label in enumerate(labels):
        pred[:, index] = (y_prob[:, index] >= thresholds[label]).astype(int)
    per_label = {}
    for index, label in enumerate(labels):
        try:
            auc = float(roc_auc_score(y_true[:, index], y_prob[:, index]))
        except ValueError:
            auc = float("nan")
        try:
            ap = float(average_precision_score(y_true[:, index], y_prob[:, index]))
        except ValueError:
            ap = float("nan")
        per_label[label] = {
            "f1": float(f1_score(y_true[:, index], pred[:, index], zero_division=0)),
            "roc_auc": auc,
            "average_precision": ap,
        }
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "mean_roc_auc": float(np.nanmean([item["roc_auc"] for item in per_label.values()])),
        "macro_average_precision": float(np.nanmean([item["average_precision"] for item in per_label.values()])),
        "per_label": per_label,
    }
```

- [ ] **Step 5: Run folds and metrics tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_folds_metrics.py -v
```

Expected:

```text
PASSED
```

- [ ] **Step 6: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final
git add Chest-Disease/src/chest_disease/folds.py Chest-Disease/src/chest_disease/metrics.py Chest-Disease/tests/test_folds_metrics.py
git commit -m "feat(chest): add folds metrics and thresholding"
```

Expected: commit succeeds.

### Task 5: Image Dataset, Transforms, And Model Factories

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/dataset.py`
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/models.py`
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_dataset_models.py`

- [ ] **Step 1: Write failing dataset and model tests**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_dataset_models.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from chest_disease.config import EXPECTED_COLUMNS, LABEL_COLUMNS, RunConfig
from chest_disease.dataset import ChestDiseaseDataset, build_transforms
from chest_disease.models import create_model


def make_image(path: Path) -> None:
    array = np.full((32, 48), 128, dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_dataset_returns_image_tensor_and_multilabel_target(tmp_path: Path):
    image_path = tmp_path / "cxr00001.jpg"
    make_image(image_path)
    frame = pd.DataFrame([["cxr00001.jpg"] + [0] * len(LABEL_COLUMNS)], columns=EXPECTED_COLUMNS)
    frame.loc[0, "Atelectasis"] = 1
    frame["image_path"] = [str(image_path)]
    transforms = build_transforms(image_size=64, train=False, model_family="timm")
    dataset = ChestDiseaseDataset(frame, transforms=transforms, include_targets=True)
    image, target, filename = dataset[0]
    assert image.shape == (3, 64, 64)
    assert target.shape == (13,)
    assert target[LABEL_COLUMNS.index("Atelectasis")] == 1
    assert filename == "cxr00001.jpg"


def test_create_timm_model_has_13_outputs():
    config = RunConfig(model_name="resnet18", allow_external_weights=False)
    model = create_model(config)
    output = model(torch.zeros(2, 3, 64, 64))
    assert output.shape == (2, 13)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_dataset_models.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'chest_disease.dataset'
```

- [ ] **Step 3: Implement dataset and transforms**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/dataset.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import ID_COLUMN, LABEL_COLUMNS


class ChestDiseaseDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transforms, include_targets: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transforms = transforms
        self.include_targets = include_targets

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(Path(row["image_path"])).convert("RGB")
        image_tensor = self.transforms(image)
        filename = str(row[ID_COLUMN])
        if self.include_targets:
            target = torch.tensor(row[LABEL_COLUMNS].astype(float).to_numpy(), dtype=torch.float32)
            return image_tensor, target, filename
        return image_tensor, filename


def build_transforms(image_size: int, train: bool, model_family: str):
    ops = []
    if train:
        ops.extend([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    else:
        ops.append(transforms.Resize((image_size, image_size)))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(ops)
```

- [ ] **Step 4: Implement model factory**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/models.py`:

```python
from __future__ import annotations

import torch
from torch import nn

from .config import LABEL_COLUMNS, RunConfig


def _create_timm_model(model_name: str, pretrained: bool) -> nn.Module:
    import timm

    return timm.create_model(model_name, pretrained=pretrained, num_classes=len(LABEL_COLUMNS))


def _create_torchxrayvision_model() -> nn.Module:
    import torchxrayvision as xrv

    backbone = xrv.models.DenseNet(weights="densenet121-res224-all")
    in_features = backbone.classifier.in_features
    backbone.classifier = nn.Linear(in_features, len(LABEL_COLUMNS))
    return backbone


def create_model(config: RunConfig) -> nn.Module:
    if config.model_name.startswith("torchxrayvision"):
        if not config.allow_external_weights:
            return _create_timm_model("resnet18", pretrained=False)
        return _create_torchxrayvision_model()
    return _create_timm_model(config.model_name, pretrained=config.allow_external_weights)
```

- [ ] **Step 5: Run dataset and model tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_dataset_models.py -v
```

Expected:

```text
PASSED
```

- [ ] **Step 6: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final
git add Chest-Disease/src/chest_disease/dataset.py Chest-Disease/src/chest_disease/models.py Chest-Disease/tests/test_dataset_models.py
git commit -m "feat(chest): add image dataset and model factories"
```

Expected: commit succeeds.

### Task 6: Training, Inference, And Artifacts

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/train.py`
- Modify: `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_dataset_models.py`

- [ ] **Step 1: Add failing training loop tests**

Append to `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_dataset_models.py`:

```python
from torch.utils.data import DataLoader, TensorDataset

from chest_disease.train import predict_logits, train_one_epoch


def test_train_one_epoch_updates_linear_model_on_cpu():
    model = torch.nn.Linear(4, 13)
    dataset = TensorDataset(torch.randn(8, 4), torch.zeros(8, 13))
    loader = DataLoader(dataset, batch_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, loader, optimizer, device=torch.device("cpu"), use_amp=False)
    assert loss >= 0.0


def test_predict_logits_returns_numpy_array():
    model = torch.nn.Linear(4, 13)
    dataset = TensorDataset(torch.randn(8, 4), torch.zeros(8, 13))
    loader = DataLoader(dataset, batch_size=4)
    logits = predict_logits(model, loader, device=torch.device("cpu"))
    assert logits.shape == (8, 13)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_dataset_models.py::test_train_one_epoch_updates_linear_model_on_cpu tests/test_dataset_models.py::test_predict_logits_returns_numpy_array -v
```

Expected:

```text
ModuleNotFoundError: No module named 'chest_disease.train'
```

- [ ] **Step 3: Implement train helpers**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/train.py`:

```python
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _batch_inputs_targets(batch):
    if len(batch) == 3:
        inputs, targets, _ = batch
    else:
        inputs, targets = batch
    return inputs, targets


def train_one_epoch(model: nn.Module, loader, optimizer, device: torch.device, use_amp: bool) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    for batch in loader:
        inputs, targets = _batch_inputs_targets(batch)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    for batch in loader:
        if len(batch) == 3:
            inputs = batch[0]
        elif len(batch) == 2 and not torch.is_tensor(batch[1]):
            inputs = batch[0]
        else:
            inputs = batch[0]
        logits = model(inputs.to(device))
        outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)
```

- [ ] **Step 4: Run training helper tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_dataset_models.py::test_train_one_epoch_updates_linear_model_on_cpu tests/test_dataset_models.py::test_predict_logits_returns_numpy_array -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final
git add Chest-Disease/src/chest_disease/train.py Chest-Disease/tests/test_dataset_models.py
git commit -m "feat(chest): add training and inference helpers"
```

Expected: commit succeeds.

### Task 7: End-To-End Pipeline And Smoke Test

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/pipeline.py`
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write failing pipeline smoke test**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_pipeline_smoke.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd

from chest_disease.config import EXPECTED_COLUMNS, LABEL_COLUMNS, ProjectPaths, RunConfig
from chest_disease.pipeline import create_submission_from_probabilities


def test_create_submission_from_probabilities_writes_valid_binary_csv(tmp_path: Path):
    test_template = pd.DataFrame(
        [
            ["cxr00001.jpg"] + [""] * len(LABEL_COLUMNS),
            ["cxr00002.jpg"] + [""] * len(LABEL_COLUMNS),
        ],
        columns=EXPECTED_COLUMNS,
    )
    template_path = tmp_path / "test_submission.csv"
    test_template.to_csv(template_path, index=False)
    paths = ProjectPaths(
        competition_dir=tmp_path,
        image_dir=tmp_path / "images",
        train_csv=tmp_path / "train.csv",
        test_submission_csv=template_path,
        working_dir=tmp_path / "working",
        submission_path=tmp_path / "submission.csv",
    )
    probabilities = np.zeros((2, len(LABEL_COLUMNS)), dtype=float)
    probabilities[0, LABEL_COLUMNS.index("Atelectasis")] = 0.8
    thresholds = {label: 0.5 for label in LABEL_COLUMNS}
    result = create_submission_from_probabilities(paths, probabilities, thresholds, RunConfig(output_mode="binary"))
    submission = pd.read_csv(paths.submission_path)
    assert result.rows == 2
    assert submission.loc[0, "Atelectasis"] == 1
    assert submission.loc[0, "No Finding"] == 0
    assert submission.loc[1, "No Finding"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_pipeline_smoke.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'chest_disease.pipeline'
```

- [ ] **Step 3: Implement pipeline submission creation**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/pipeline.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABEL_COLUMNS, ProjectPaths, RunConfig
from .data import validate_submission
from .metrics import apply_no_finding_consistency


def create_submission_from_probabilities(
    paths: ProjectPaths,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
    config: RunConfig,
):
    template = pd.read_csv(paths.test_submission_csv)
    if probabilities.shape != (len(template), len(LABEL_COLUMNS)):
        raise ValueError(f"probability shape {probabilities.shape} does not match test rows and labels")
    output = template.copy()
    if config.output_mode == "probability":
        output[LABEL_COLUMNS] = probabilities
    elif config.output_mode == "binary":
        binary = pd.DataFrame(probabilities, columns=LABEL_COLUMNS)
        for label in LABEL_COLUMNS:
            binary[label] = (binary[label] >= thresholds[label]).astype(int)
        output[LABEL_COLUMNS] = apply_no_finding_consistency(binary).astype(int)
    else:
        raise ValueError("output_mode must be 'binary' or 'probability'")
    paths.submission_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(paths.submission_path, index=False)
    return validate_submission(paths.submission_path, paths.test_submission_csv, config.output_mode)
```

- [ ] **Step 4: Run smoke test**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_pipeline_smoke.py -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Extend pipeline for real training orchestration**

Add these functions to `/Users/temicide/Documents/5_domain_final/Chest-Disease/src/chest_disease/pipeline.py`:

```python
def ensure_working_dirs(paths: ProjectPaths) -> None:
    for name in ["oof", "test_preds", "thresholds", "submissions", "checkpoints", "logs"]:
        (paths.working_dir / name).mkdir(parents=True, exist_ok=True)


def save_probabilities(path: Path, filenames: list[str], probabilities: np.ndarray) -> None:
    frame = pd.DataFrame(probabilities, columns=LABEL_COLUMNS)
    frame.insert(0, "filename", filenames)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_probability_csv(path: Path) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(path)
    return frame["filename"].astype(str).tolist(), frame[LABEL_COLUMNS].astype(float).to_numpy()


def run_colab_pipeline(paths: ProjectPaths, config: RunConfig):
    ensure_working_dirs(paths)
    test_template = pd.read_csv(paths.test_submission_csv)
    prevalence = np.zeros((len(test_template), len(LABEL_COLUMNS)), dtype=float)
    prevalence[:, LABEL_COLUMNS.index("No Finding")] = 1.0
    thresholds = {label: 0.5 for label in LABEL_COLUMNS}
    save_probabilities(paths.working_dir / "test_preds" / "prevalence_baseline.csv", test_template["filename"].astype(str).tolist(), prevalence)
    return create_submission_from_probabilities(paths, prevalence, thresholds, config)
```

This `run_colab_pipeline` is a valid end-to-end baseline that writes a complete CSV. In the next implementation pass, replace the prevalence probability block with fold training and test inference while preserving the same function signature and artifact paths.

- [ ] **Step 6: Run all tests**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest -v
```

Expected:

```text
PASSED
```

- [ ] **Step 7: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final
git add Chest-Disease/src/chest_disease/pipeline.py Chest-Disease/tests/test_pipeline_smoke.py
git commit -m "feat(chest): add submission pipeline"
```

Expected: commit succeeds.

### Task 8: Colab Notebook Builder

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/scripts/build_colab_notebook.py`
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/notebooks/chest_disease_colab_a100_solution.ipynb`
- Modify: `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_pipeline_smoke.py`

- [ ] **Step 1: Add failing notebook builder test**

Append to `/Users/temicide/Documents/5_domain_final/Chest-Disease/tests/test_pipeline_smoke.py`:

```python
import nbformat

from scripts.build_colab_notebook import build_notebook


def test_build_colab_notebook_contains_required_cells(tmp_path: Path):
    output = tmp_path / "solution.ipynb"
    build_notebook(output)
    notebook = nbformat.read(output, as_version=4)
    joined = "\n".join(cell.get("source", "") for cell in notebook.cells)
    assert "pip install -q kaggle timm torchxrayvision iterative-stratification" in joined
    assert "Path('/content/Chest-Disease/src/chest_disease/config.py')" in joined
    assert "write_text" in joined
    assert "kaggle competitions download -c chest-disease-detection" in joined
    assert "/content/submission.csv" in joined
    assert "kaggle competitions submit" not in joined
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_pipeline_smoke.py::test_build_colab_notebook_contains_required_cells -v
```

Expected:

```text
ModuleNotFoundError: No module named 'scripts'
```

- [ ] **Step 3: Implement notebook builder**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/scripts/build_colab_notebook.py`:

```python
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path("/Users/temicide/Documents/5_domain_final/Chest-Disease")
COLAB_PROJECT_ROOT = Path("/content/Chest-Disease")
SOURCE_FILES = [
    "src/chest_disease/__init__.py",
    "src/chest_disease/config.py",
    "src/chest_disease/kaggle_io.py",
    "src/chest_disease/data.py",
    "src/chest_disease/folds.py",
    "src/chest_disease/metrics.py",
    "src/chest_disease/dataset.py",
    "src/chest_disease/models.py",
    "src/chest_disease/train.py",
    "src/chest_disease/pipeline.py",
]


def source_file_cells() -> list:
    cells = []
    for relative in SOURCE_FILES:
        source = (PROJECT_ROOT / relative).read_text()
        colab_path = COLAB_PROJECT_ROOT / relative
        cells.append(
            nbf.v4.new_code_cell(
                "from pathlib import Path\n"
                f"path = Path({str(colab_path)!r})\n"
                "path.parent.mkdir(parents=True, exist_ok=True)\n"
                f"path.write_text({source!r})\n"
            )
        )
    return cells


def build_notebook(output_path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Chest Disease Detection Colab A100 Solution"),
        nbf.v4.new_code_cell("!pip install -q kaggle timm torchxrayvision iterative-stratification nbformat"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import os, zipfile, subprocess\n"
            "INPUT_ROOT = Path('/content/input')\n"
            "INPUT_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "WORKING = Path('/content/working')\n"
            "WORKING.mkdir(parents=True, exist_ok=True)\n"
        ),
        nbf.v4.new_code_cell(
            "from google.colab import files\n"
            "try:\n"
            "    from google.colab import userdata\n"
            "    if not os.environ.get('KAGGLE_USERNAME'):\n"
            "        os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME') or ''\n"
            "    if not os.environ.get('KAGGLE_KEY'):\n"
            "        os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY') or ''\n"
            "except Exception:\n"
            "    pass\n"
            "kaggle_json = Path('/root/.kaggle/kaggle.json')\n"
            "if not (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')) and not kaggle_json.exists():\n"
            "    uploaded = files.upload()\n"
            "    if 'kaggle.json' not in uploaded:\n"
            "        raise RuntimeError('Upload kaggle.json or set KAGGLE_USERNAME/KAGGLE_KEY')\n"
            "    kaggle_json.parent.mkdir(parents=True, exist_ok=True)\n"
            "    kaggle_json.write_bytes(uploaded['kaggle.json'])\n"
            "    kaggle_json.chmod(0o600)\n"
        ),
        nbf.v4.new_code_cell(
            "!kaggle competitions download -c chest-disease-detection -p /content/input\n"
            "archive = Path('/content/input/chest-disease-detection.zip')\n"
            "with zipfile.ZipFile(archive) as zf:\n"
            "    zf.extractall('/content/input')\n"
        ),
        *source_file_cells(),
        nbf.v4.new_code_cell(
            "import sys\n"
            "sys.path.insert(0, '/content/Chest-Disease/src')\n"
            "from chest_disease.config import RunConfig\n"
            "from chest_disease.kaggle_io import resolve_project_paths\n"
            "from chest_disease.pipeline import run_colab_pipeline\n"
            "paths = resolve_project_paths()\n"
            "result = run_colab_pipeline(paths, RunConfig())\n"
            "print(f'Wrote /content/submission.csv with {result.rows} rows')\n"
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output_path)


if __name__ == '__main__':
    build_notebook(Path('/Users/temicide/Documents/5_domain_final/Chest-Disease/notebooks/chest_disease_colab_a100_solution.ipynb'))
```

- [ ] **Step 4: Generate notebook**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 scripts/build_colab_notebook.py
```

Expected: creates `/Users/temicide/Documents/5_domain_final/Chest-Disease/notebooks/chest_disease_colab_a100_solution.ipynb`.

- [ ] **Step 5: Run notebook builder test**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest tests/test_pipeline_smoke.py::test_build_colab_notebook_contains_required_cells -v
```

Expected:

```text
PASSED
```

- [ ] **Step 6: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final
git add Chest-Disease/scripts/build_colab_notebook.py Chest-Disease/notebooks/chest_disease_colab_a100_solution.ipynb Chest-Disease/tests/test_pipeline_smoke.py
git commit -m "feat(chest): add colab notebook builder"
```

Expected: commit succeeds.

### Task 9: Final Local Verification And Submission Log

**Files:**
- Create: `/Users/temicide/Documents/5_domain_final/Chest-Disease/submission_log.md`
- Modify: files changed by earlier tasks only if tests expose a defect.

- [ ] **Step 1: Create submission log**

Create `/Users/temicide/Documents/5_domain_final/Chest-Disease/submission_log.md`:

```markdown
# Chest Disease Submission Log

Competition: https://www.kaggle.com/competitions/chest-disease-detection/data

Manual submissions only. Do not run `kaggle competitions submit`.

| Date | Candidate | CV mean ROC-AUC | CV macro F1 | Public LB | Notes |
|---|---|---:|---:|---:|---|
| 2026-06-06 | scaffold-only | n/a | n/a | n/a | Pipeline and notebook scaffold created. |
```

- [ ] **Step 2: Run full test suite**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final/Chest-Disease
python3 -m pytest -v
```

Expected:

```text
PASSED
```

- [ ] **Step 3: Check for forbidden automated submission commands**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final
rg -n "kaggle competitions submit|competition_submit|api.competition_submit" Chest-Disease || true
```

Expected: no matches except text that explicitly says the command is forbidden.

- [ ] **Step 4: Check generated notebook path and spec coverage**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final
test -f Chest-Disease/notebooks/chest_disease_colab_a100_solution.ipynb
test -f Chest-Disease/src/chest_disease/pipeline.py
test -f Chest-Disease/submission_log.md
```

Expected: command exits with status `0`.

- [ ] **Step 5: Commit**

Run:

```bash
cd /Users/temicide/Documents/5_domain_final
git add Chest-Disease/submission_log.md
git commit -m "docs(chest): add submission log"
```

Expected: commit succeeds.

## Self-Review

Spec coverage:

- Colab A100 notebook: Task 8.
- Secure Kaggle credential handling: Task 2.
- Kaggle download before data loading: Task 2 and Task 8.
- Local development fallbacks: Task 1 and Task 2.
- Data audit schema and image path checks: Task 3.
- Multi-label classification instead of softmax: Task 5 and Task 6.
- Multi-label folds: Task 4.
- Metrics and threshold tuning: Task 4.
- `No Finding` consistency: Task 4 and Task 7.
- A100 mixed precision: Task 1 and Task 6.
- External pretrained weight gate: Task 1 and Task 5.
- Submission validation and no automated submit: Task 3, Task 7, Task 8, and Task 9.
- Experiment artifacts: Task 1 and Task 7.

Placeholder scan:

- The plan uses concrete task steps, explicit code blocks, exact commands, and named functions defined in prior tasks.
- The notebook builder imports and runs `run_colab_pipeline`, which is defined in Task 7 and writes a valid baseline CSV before stronger training is added.

Type consistency:

- `RunConfig`, `ProjectPaths`, `LABEL_COLUMNS`, `EXPECTED_COLUMNS`, `create_submission_from_probabilities`, and `validate_submission` names are consistent across tasks.
- `output_mode` values are consistently `binary` or `probability`.
- `filename` remains the ID column in every task.
