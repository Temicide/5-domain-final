from __future__ import annotations

from pathlib import Path

import pandas as pd


class SubmissionValidationError(ValueError):
    pass


def load_sample_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"file_name": "string", "text": "string"}, keep_default_na=False)
    if list(df.columns) != ["file_name", "text"]:
        raise SubmissionValidationError(
            f"Expected sample submission columns ['file_name', 'text'], got {list(df.columns)}"
        )
    if df["file_name"].duplicated().any():
        duplicates = df.loc[df["file_name"].duplicated(), "file_name"].tolist()
        raise SubmissionValidationError(f"Duplicate file_name values: {', '.join(duplicates[:10])}")
    return df


def validate_audio_coverage(sample_df: pd.DataFrame, audio_dir: Path) -> None:
    missing = [name for name in sample_df["file_name"].tolist() if not (audio_dir / name).is_file()]
    if missing:
        raise SubmissionValidationError(f"Missing audio files: {', '.join(missing[:20])}")


def validate_submission_frame(submission_df: pd.DataFrame, allow_empty_files: set[str] | None = None) -> None:
    allow_empty_files = allow_empty_files or set()
    if list(submission_df.columns) != ["file_name", "text"]:
        raise SubmissionValidationError(
            f"Expected submission columns ['file_name', 'text'], got {list(submission_df.columns)}"
        )
    if submission_df["file_name"].duplicated().any():
        duplicates = submission_df.loc[submission_df["file_name"].duplicated(), "file_name"].tolist()
        raise SubmissionValidationError(f"Duplicate submission rows: {', '.join(duplicates[:10])}")
    for row in submission_df.itertuples(index=False):
        text = "" if pd.isna(row.text) else str(row.text)
        if text == "" and row.file_name not in allow_empty_files:
            raise SubmissionValidationError(f"Empty transcript for {row.file_name}")


def write_submission_csv(
    sample_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_path: Path,
    allow_empty_files: set[str] | None = None,
) -> pd.DataFrame:
    required_prediction_columns = {"file_name", "normalized_text"}
    missing_columns = required_prediction_columns - set(predictions_df.columns)
    if missing_columns:
        raise SubmissionValidationError(f"Prediction columns missing: {', '.join(sorted(missing_columns))}")

    merged = sample_df[["file_name"]].merge(
        predictions_df[["file_name", "normalized_text"]],
        on="file_name",
        how="left",
        validate="one_to_one",
    )
    if merged["normalized_text"].isna().any():
        missing = merged.loc[merged["normalized_text"].isna(), "file_name"].tolist()
        raise SubmissionValidationError(f"Missing predictions: {', '.join(missing[:20])}")

    submission_df = merged.rename(columns={"normalized_text": "text"})[["file_name", "text"]]
    submission_df["text"] = submission_df["text"].fillna("").astype(str)
    validate_submission_frame(submission_df, allow_empty_files=allow_empty_files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return submission_df
