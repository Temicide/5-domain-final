from pathlib import Path

import pandas as pd
import pytest

from math_vqa.submission import (
    PredictionRecord,
    build_submission_frames,
    validate_submission,
    write_outputs,
)


def test_build_submission_frames_preserves_sample_order() -> None:
    sample_df = pd.DataFrame({"id": ["2", "1"], "answer": ["2", "2"]})
    records = [
        PredictionRecord("1", "images/1.jpg", "raw one", "1", "base", "raw", "640x480", 1.2, "", False),
        PredictionRecord("2", "images/2.jpg", "raw two", "2", "diagram", "high_res", "1024x768", 2.4, "", False),
    ]

    submission_df, raw_df = build_submission_frames(records, sample_df)

    assert submission_df.to_dict("list") == {"id": ["2", "1"], "answer": ["2", "1"]}
    assert list(raw_df.columns) == [
        "id",
        "image_path",
        "raw_prediction",
        "clean_answer",
        "prompt_name",
        "preprocess_name",
        "final_size",
        "runtime_seconds",
        "inference_error",
        "used_fallback",
    ]


def test_validate_submission_rejects_empty_answer() -> None:
    sample_df = pd.DataFrame({"id": ["1"], "answer": ["2"]})
    submission_df = pd.DataFrame({"id": ["1"], "answer": [" "]})

    with pytest.raises(ValueError, match="empty answers"):
        validate_submission(sample_df, submission_df)


def test_validate_submission_rejects_wrong_order() -> None:
    sample_df = pd.DataFrame({"id": ["1", "2"], "answer": ["2", "2"]})
    submission_df = pd.DataFrame({"id": ["2", "1"], "answer": ["2", "1"]})

    with pytest.raises(ValueError, match="row order"):
        validate_submission(sample_df, submission_df)


def test_write_outputs_writes_submission_and_raw_predictions(tmp_path: Path) -> None:
    sample_df = pd.DataFrame({"id": ["1"], "answer": ["2"]})
    records = [PredictionRecord("1", "images/1.jpg", "raw", "7", "base", "raw", "20x10", 0.5, "", False)]

    submission_path, raw_path = write_outputs(records, sample_df, tmp_path)

    assert submission_path == tmp_path / "submission.csv"
    assert raw_path == tmp_path / "raw_predictions.csv"
    assert submission_path.read_text(encoding="utf-8").startswith("id,answer\n1,7\n")
    assert "raw_prediction" in raw_path.read_text(encoding="utf-8")
