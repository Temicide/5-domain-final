from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sleep_stage.config import EPOCH_ROWS, LABELS, LABEL_COLUMN, SIGNAL_COLUMNS


@dataclass(frozen=True)
class SubmissionValidation:
    rows: int
    valid_labels: list[str]
    id_match: bool


def validate_submission(submission_path: Path, sample_submission_path: Path) -> SubmissionValidation:
    submission = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_submission_path)
    expected_columns = ["id", "labels"]
    if list(submission.columns) != expected_columns:
        raise ValueError(f"submission columns must be {expected_columns}, found {list(submission.columns)}")
    if len(submission) != len(sample):
        raise ValueError(f"submission row count {len(submission)} does not match sample row count {len(sample)}")
    id_match = submission["id"].astype(str).tolist() == sample["id"].astype(str).tolist()
    if not id_match:
        raise ValueError("submission ids must exactly match sample_submission.csv order")
    if submission["labels"].isna().any():
        raise ValueError("submission labels contain missing values")
    invalid = sorted(set(submission["labels"].astype(str)) - set(LABELS))
    if invalid:
        raise ValueError(f"submission contains labels outside {LABELS}: {invalid}")
    return SubmissionValidation(
        rows=len(submission),
        valid_labels=submission["labels"].astype(str).tolist(),
        id_match=True,
    )


def _validate_signal_columns(frame: pd.DataFrame, require_label: bool) -> None:
    missing = [column for column in SIGNAL_COLUMNS if column not in frame.columns]
    if require_label and LABEL_COLUMN not in frame.columns:
        missing.append(LABEL_COLUMN)
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def epoch_dataframe_from_recording(frame: pd.DataFrame, recording_id: str) -> pd.DataFrame:
    _validate_signal_columns(frame, require_label=True)
    usable_rows = (len(frame) // EPOCH_ROWS) * EPOCH_ROWS
    if usable_rows == 0:
        raise ValueError(f"recording {recording_id} has no complete {EPOCH_ROWS}-row epochs")

    rows: list[dict[str, object]] = []
    for epoch_index, start in enumerate(range(0, usable_rows, EPOCH_ROWS)):
        epoch = frame.iloc[start : start + EPOCH_ROWS]
        labels = epoch[LABEL_COLUMN].astype(str)
        mode = labels.mode()
        label = mode.iloc[0] if not mode.empty else labels.iloc[0]
        rows.append(
            {
                "recording_id": recording_id,
                "epoch_index": epoch_index,
                "start_row": start,
                "end_row": start + EPOCH_ROWS,
                "label": label,
                "signals": epoch[SIGNAL_COLUMNS].astype(float).to_numpy(copy=True),
            }
        )
    return pd.DataFrame(rows)


def load_train_recording(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return epoch_dataframe_from_recording(frame, path.stem)


def iter_train_recordings(train_dir: Path):
    for path in sorted(train_dir.glob("*.csv")):
        yield load_train_recording(path)


def list_test_segment_paths(test_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(test_root.glob("test*/test*.csv")):
        paths[path.stem] = path
    return paths


def read_test_segments(sample_submission_path: Path, segment_paths: dict[str, Path]) -> pd.DataFrame:
    sample = pd.read_csv(sample_submission_path)
    rows: list[dict[str, object]] = []
    for row_index, segment_id in enumerate(sample["id"].astype(str)):
        if segment_id not in segment_paths:
            raise FileNotFoundError(f"missing test segment for sample id {segment_id}")
        frame = pd.read_csv(segment_paths[segment_id])
        _validate_signal_columns(frame, require_label=False)
        if len(frame) != EPOCH_ROWS:
            raise ValueError(f"test segment {segment_id} must contain exactly {EPOCH_ROWS} rows, found {len(frame)}")
        subject_id = segment_id.split("_")[0]
        try:
            epoch_index = int(segment_id.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            epoch_index = row_index
        rows.append(
            {
                "id": segment_id,
                "recording_id": subject_id,
                "epoch_index": epoch_index,
                "signals": frame[SIGNAL_COLUMNS].astype(float).to_numpy(copy=True),
            }
        )
    return pd.DataFrame(rows)


def write_submission_from_labels(
    labels_by_id: dict[str, str],
    sample_submission_path: Path,
    output_path: Path,
) -> Path:
    sample = pd.read_csv(sample_submission_path)
    labels = []
    for segment_id in sample["id"].astype(str):
        if segment_id not in labels_by_id:
            raise ValueError(f"missing prediction for sample id {segment_id}")
        labels.append(labels_by_id[segment_id])
    submission = pd.DataFrame({"id": sample["id"].astype(str), "labels": labels})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    validate_submission(output_path, sample_submission_path)
    return output_path


def as_numeric_signal_array(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (EPOCH_ROWS, len(SIGNAL_COLUMNS)):
        raise ValueError(f"signal array must have shape {(EPOCH_ROWS, len(SIGNAL_COLUMNS))}, found {array.shape}")
    return array
