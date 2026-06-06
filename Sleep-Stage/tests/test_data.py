from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleep_stage.config import (
    COLAB_INPUT_ROOT,
    COLAB_OUTPUT_PATH,
    COMPETITION_SLUG,
    EPOCH_ROWS,
    LABELS,
    SIGNAL_COLUMNS,
    build_paths,
    find_competition_root,
)
from sleep_stage.data import (
    epoch_dataframe_from_recording,
    list_test_segment_paths,
    read_test_segments,
    validate_submission,
    write_submission_from_labels,
)
from sleep_stage.kaggle_download import verify_extracted_competition_data


def _signals(rows: int) -> pd.DataFrame:
    values = np.arange(rows, dtype=float)
    return pd.DataFrame({column: values + index for index, column in enumerate(SIGNAL_COLUMNS)})


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


def test_verify_extracted_competition_data_requires_expected_files(tmp_path: Path):
    root = tmp_path / COMPETITION_SLUG
    (root / "train" / "train").mkdir(parents=True)
    (root / "test_segment" / "test_segment" / "test001").mkdir(parents=True)
    (root / "sample_submission.csv").write_text("id,labels\n")

    verified = verify_extracted_competition_data(root)

    assert verified == root


def test_epoch_dataframe_from_recording_returns_one_row_per_480_samples():
    frame = _signals(EPOCH_ROWS * 2)
    frame["Sleep_Stage"] = ["W"] * EPOCH_ROWS + ["N2"] * EPOCH_ROWS

    epochs = epoch_dataframe_from_recording(frame, "train001")

    assert len(epochs) == 2
    assert epochs["label"].tolist() == ["W", "N2"]
    assert epochs.loc[0, "signals"].shape == (EPOCH_ROWS, len(SIGNAL_COLUMNS))


def test_list_and_read_test_segments_follow_sample_submission_order(tmp_path: Path):
    test_root = tmp_path / "test_segment" / "test_segment"
    first = test_root / "test001"
    second = test_root / "test002"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _signals(EPOCH_ROWS).to_csv(first / "test001_00000.csv", index=False)
    _signals(EPOCH_ROWS).to_csv(second / "test002_00000.csv", index=False)
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": ["test002_00000", "test001_00000"], "labels": ["N2", "N2"]}).to_csv(sample_path, index=False)

    segment_paths = list_test_segment_paths(test_root)
    table = read_test_segments(sample_path, segment_paths)

    assert table["id"].tolist() == ["test002_00000", "test001_00000"]
    assert table.loc[0, "signals"].shape == (EPOCH_ROWS, len(SIGNAL_COLUMNS))


def test_read_test_segments_requires_exact_epoch_rows(tmp_path: Path):
    test_root = tmp_path / "test_segment" / "test_segment" / "test001"
    test_root.mkdir(parents=True)
    _signals(EPOCH_ROWS - 1).to_csv(test_root / "test001_00000.csv", index=False)
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": ["test001_00000"], "labels": ["N2"]}).to_csv(sample_path, index=False)

    with pytest.raises(ValueError, match="exactly"):
        read_test_segments(sample_path, list_test_segment_paths(test_root.parent))


def test_write_submission_from_labels_preserves_sample_order_and_validates(tmp_path: Path):
    sample_path = tmp_path / "sample_submission.csv"
    output_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["test001_00001", "test001_00000"], "labels": ["N2", "N2"]}).to_csv(sample_path, index=False)

    written = write_submission_from_labels({"test001_00000": "W", "test001_00001": "R"}, sample_path, output_path)

    assert written == output_path
    submission = pd.read_csv(output_path)
    assert list(submission.columns) == ["id", "labels"]
    assert submission["id"].tolist() == ["test001_00001", "test001_00000"]
    assert submission["labels"].tolist() == ["R", "W"]
