from pathlib import Path

import pandas as pd
import pytest

from call_asr.submission import (
    SubmissionValidationError,
    load_sample_submission,
    validate_audio_coverage,
    validate_submission_frame,
    write_submission_csv,
)


def test_load_sample_submission_preserves_schema_and_order(tmp_path):
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("file_name,text\nRSP_002_audio.wav,\nRSP_001_audio.wav,\n", encoding="utf-8")

    df = load_sample_submission(sample)

    assert list(df.columns) == ["file_name", "text"]
    assert df["file_name"].tolist() == ["RSP_002_audio.wav", "RSP_001_audio.wav"]


def test_validate_audio_coverage_reports_missing_file(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "RSP_001_audio.wav").write_bytes(b"fake")
    df = pd.DataFrame({"file_name": ["RSP_001_audio.wav", "SDB_001_audio.wav"], "text": ["", ""]})

    with pytest.raises(SubmissionValidationError, match="Missing audio files: SDB_001_audio.wav"):
        validate_audio_coverage(df, audio_dir)


def test_validate_submission_frame_rejects_empty_non_silence_text():
    df = pd.DataFrame({"file_name": ["RSP_001_audio.wav"], "text": [""]})

    with pytest.raises(SubmissionValidationError, match="Empty transcript for RSP_001_audio.wav"):
        validate_submission_frame(df, allow_empty_files=set())


def test_write_submission_csv_preserves_sample_order(tmp_path):
    output = tmp_path / "submission.csv"
    sample_df = pd.DataFrame({"file_name": ["RSP_002_audio.wav", "RSP_001_audio.wav"], "text": ["", ""]})
    predictions = pd.DataFrame(
        {
            "file_name": ["RSP_001_audio.wav", "RSP_002_audio.wav"],
            "normalized_text": ["หนึ่ง", "สอง"],
        }
    )

    write_submission_csv(sample_df, predictions, output)

    written = pd.read_csv(output)
    assert written.to_dict("records") == [
        {"file_name": "RSP_002_audio.wav", "text": "สอง"},
        {"file_name": "RSP_001_audio.wav", "text": "หนึ่ง"},
    ]
