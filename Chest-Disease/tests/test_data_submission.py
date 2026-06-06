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
    root.mkdir(parents=True, exist_ok=True)
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
