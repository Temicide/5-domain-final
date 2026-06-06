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
