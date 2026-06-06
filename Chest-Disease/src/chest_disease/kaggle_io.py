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
        COLAB_COMPETITION_DIR,
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
