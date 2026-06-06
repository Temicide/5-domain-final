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
        target.write_text(json.dumps({"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]}))
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
