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
