from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PredictionRecord:
    id: str
    image_path: str
    raw_prediction: str
    clean_answer: str
    prompt_name: str
    preprocess_name: str
    final_size: str
    runtime_seconds: float
    inference_error: str
    used_fallback: bool


def build_submission_frames(
    records: list[PredictionRecord],
    sample_submission: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    record_by_id = {str(record.id): record for record in records}
    sample_ids = [str(image_id) for image_id in sample_submission["id"].tolist()]
    missing_ids = [image_id for image_id in sample_ids if image_id not in record_by_id]
    if missing_ids:
        raise ValueError(f"missing predictions for ids: {missing_ids[:10]}")

    submission_rows = [
        {"id": image_id, "answer": str(record_by_id[image_id].clean_answer)}
        for image_id in sample_ids
    ]
    raw_rows = [asdict(record) for record in records]
    submission_df = pd.DataFrame(submission_rows, columns=["id", "answer"])
    raw_df = pd.DataFrame(
        raw_rows,
        columns=[
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
        ],
    )
    return submission_df, raw_df


def validate_submission(sample_submission: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["id", "answer"]:
        raise ValueError(f"submission columns must be exactly ['id', 'answer'], got {list(submission.columns)}")
    if len(submission) != len(sample_submission):
        raise ValueError(f"submission row count must be {len(sample_submission)}, got {len(submission)}")

    sample_ids = [str(value) for value in sample_submission["id"].tolist()]
    submission_ids = [str(value) for value in submission["id"].tolist()]
    if submission_ids != sample_ids:
        raise ValueError("submission row order must match sample_submission.csv")
    if len(set(submission_ids)) != len(submission_ids):
        raise ValueError("submission ids must be unique")
    if submission["answer"].isna().any():
        raise ValueError("submission has null answers")

    empty_mask = submission["answer"].astype(str).str.strip().eq("")
    if empty_mask.any():
        empty_ids = submission.loc[empty_mask, "id"].astype(str).tolist()
        raise ValueError(f"submission has empty answers for ids: {empty_ids[:10]}")


def write_outputs(
    records: list[PredictionRecord],
    sample_submission: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_df, raw_df = build_submission_frames(records, sample_submission)
    validate_submission(sample_submission, submission_df)

    submission_path = output_dir / "submission.csv"
    raw_path = output_dir / "raw_predictions.csv"
    submission_df.to_csv(submission_path, index=False)
    raw_df.to_csv(raw_path, index=False)
    return submission_path, raw_path
