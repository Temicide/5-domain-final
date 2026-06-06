from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import EXPECTED_COLUMNS, ID_COLUMN, LABEL_COLUMNS


@dataclass(frozen=True)
class SubmissionValidation:
    rows: int
    columns_match: bool
    filenames_match: bool
    no_missing_labels: bool
    values_in_range: bool


def _assert_columns(df: pd.DataFrame, source: Path) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{source} columns do not match expected competition schema")


def load_competition_frames(train_csv: Path, test_submission_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_submission_csv)
    _assert_columns(train, train_csv)
    _assert_columns(test, test_submission_csv)
    train[LABEL_COLUMNS] = train[LABEL_COLUMNS].astype(float)
    return train, test


def resolve_image_paths(frame: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    resolved = frame.copy()
    paths = []
    missing = []
    for filename in resolved[ID_COLUMN].astype(str):
        path = image_dir / filename
        paths.append(str(path))
        if not path.exists():
            missing.append(filename)
    if missing:
        raise FileNotFoundError("Missing image files: " + ", ".join(missing[:10]))
    resolved["image_path"] = paths
    return resolved


def validate_submission(submission_path: Path, template_path: Path, output_mode: str) -> SubmissionValidation:
    submission = pd.read_csv(submission_path)
    template = pd.read_csv(template_path)
    columns_match = list(submission.columns) == list(template.columns) == EXPECTED_COLUMNS
    if not columns_match:
        raise ValueError("submission columns do not match test_submission.csv")
    filenames_match = submission[ID_COLUMN].astype(str).tolist() == template[ID_COLUMN].astype(str).tolist()
    if not filenames_match:
        raise ValueError("submission filenames do not match test_submission.csv order")
    labels = submission[LABEL_COLUMNS]
    no_missing = not labels.isna().any().any()
    if not no_missing:
        raise ValueError("submission has missing label values")
    numeric = labels.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("submission has non-numeric label values")
    if output_mode == "binary":
        values_ok = numeric.isin([0, 1]).all().all()
    elif output_mode == "probability":
        values_ok = ((numeric >= 0.0) & (numeric <= 1.0)).all().all()
    else:
        raise ValueError("output_mode must be 'binary' or 'probability'")
    if not values_ok:
        raise ValueError(f"submission label values are invalid for {output_mode} mode")
    return SubmissionValidation(
        rows=len(submission),
        columns_match=bool(columns_match),
        filenames_match=bool(filenames_match),
        no_missing_labels=bool(no_missing),
        values_in_range=bool(values_ok),
    )
