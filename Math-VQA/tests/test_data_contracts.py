from pathlib import Path

import pandas as pd
import pytest

from math_vqa.data import (
    CompetitionPaths,
    answer_prior,
    load_competition_frames,
    resolve_image_path,
    validate_data_files,
)


def make_competition_root(tmp_path: Path) -> CompetitionPaths:
    root = tmp_path / "competition"
    (root / "images" / "images").mkdir(parents=True)
    (root / "images" / "images" / "0.jpg").write_bytes(b"image-0")
    (root / "images" / "images" / "1.jpg").write_bytes(b"image-1")
    (root / "images" / "images" / "2.jpg").write_bytes(b"image-2")
    (root / "train.csv").write_text(
        "id,image_path,answer\n0,images/0.jpg,๒\n2,images/2.jpg,2\n",
        encoding="utf-8",
    )
    (root / "test.csv").write_text(
        "id,image_path\n1,images/1.jpg\n",
        encoding="utf-8",
    )
    (root / "sample_submission.csv").write_text(
        "id,answer\n1,2\n",
        encoding="utf-8",
    )
    return CompetitionPaths(root)


def test_load_competition_frames_casts_ids_to_strings(tmp_path: Path) -> None:
    paths = make_competition_root(tmp_path)

    train_df, test_df, sample_df = load_competition_frames(paths)

    assert train_df.loc[0, "id"] == "0"
    assert test_df.loc[0, "id"] == "1"
    assert sample_df.loc[0, "id"] == "1"


def test_validate_data_files_accepts_expected_contract(tmp_path: Path) -> None:
    paths = make_competition_root(tmp_path)
    train_df, test_df, sample_df = load_competition_frames(paths)

    validate_data_files(paths, train_df, test_df, sample_df)


def test_validate_data_files_rejects_missing_image(tmp_path: Path) -> None:
    paths = make_competition_root(tmp_path)
    (paths.root / "images" / "images" / "1.jpg").unlink()
    train_df, test_df, sample_df = load_competition_frames(paths)

    with pytest.raises(ValueError, match="missing image"):
        validate_data_files(paths, train_df, test_df, sample_df)


def test_resolve_image_path_matches_nested_kaggle_image_dir(tmp_path: Path) -> None:
    paths = make_competition_root(tmp_path)

    resolved = resolve_image_path(paths, "images/1.jpg")

    assert resolved == paths.root / "images" / "images" / "1.jpg"


def test_answer_prior_prefers_most_common_simple_answer() -> None:
    train_df = pd.DataFrame({"answer": ["20 ตารางเซนติเมตร", "๒", "2", "คำตอบ"]})

    assert answer_prior(train_df) == "2"
