from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


COMPETITION_DIR_NAME = "super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen"
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


@dataclass(frozen=True)
class CompetitionPaths:
    root: Path

    @property
    def train_csv(self) -> Path:
        return self.root / "train.csv"

    @property
    def test_csv(self) -> Path:
        return self.root / "test.csv"

    @property
    def sample_submission_csv(self) -> Path:
        return self.root / "sample_submission.csv"

    @classmethod
    def auto(cls) -> "CompetitionPaths":
        kaggle_root = Path("/kaggle/input/competitions") / COMPETITION_DIR_NAME
        if kaggle_root.exists():
            return cls(kaggle_root)
        local_root = Path(__file__).resolve().parents[2] / "data" / COMPETITION_DIR_NAME
        return cls(local_root)


def load_competition_frames(paths: CompetitionPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(
        paths.train_csv,
        dtype={"id": str, "image_path": str, "answer": str},
        keep_default_na=False,
    )
    test_df = pd.read_csv(
        paths.test_csv,
        dtype={"id": str, "image_path": str},
        keep_default_na=False,
    )
    sample_df = pd.read_csv(
        paths.sample_submission_csv,
        dtype={"id": str, "answer": str},
        keep_default_na=False,
    )
    return train_df, test_df, sample_df


def resolve_image_path(paths: CompetitionPaths, image_path: str | Path) -> Path:
    image_path = Path(image_path)
    if image_path.is_absolute() and image_path.exists():
        return image_path
    candidates = [
        paths.root / image_path,
        paths.root / "images" / image_path,
        paths.root / "images" / "images" / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_data_files(
    paths: CompetitionPaths,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> None:
    if not paths.root.exists():
        raise ValueError(f"data root does not exist: {paths.root}")
    for csv_path in (paths.train_csv, paths.test_csv, paths.sample_submission_csv):
        if not csv_path.exists():
            raise ValueError(f"missing csv file: {csv_path}")

    expected_columns = {
        "train.csv": (train_df, ["id", "image_path", "answer"]),
        "test.csv": (test_df, ["id", "image_path"]),
        "sample_submission.csv": (sample_df, ["id", "answer"]),
    }
    for name, (frame, columns) in expected_columns.items():
        if list(frame.columns) != columns:
            raise ValueError(f"{name} columns must be exactly {columns}, got {list(frame.columns)}")
        if frame["id"].duplicated().any():
            duplicated = frame.loc[frame["id"].duplicated(), "id"].tolist()
            raise ValueError(f"{name} has duplicate ids: {duplicated[:5]}")

    if set(train_df["id"]) & set(test_df["id"]):
        raise ValueError("train and test ids must not overlap")
    if set(sample_df["id"]) != set(test_df["id"]):
        raise ValueError("sample_submission ids must match test ids")

    missing_images: list[str] = []
    for frame_name, frame in (("train.csv", train_df), ("test.csv", test_df)):
        for row in frame.itertuples(index=False):
            resolved = resolve_image_path(paths, row.image_path)
            if not resolved.exists():
                missing_images.append(f"{frame_name}:{row.id}:{row.image_path}")
    if missing_images:
        raise ValueError(f"missing image files: {missing_images[:10]}")


def answer_prior(train_df: pd.DataFrame) -> str:
    answers = [str(answer).strip() for answer in train_df["answer"].tolist() if str(answer).strip()]
    simple_answers = [
        answer.translate(THAI_DIGITS)
        for answer in answers
        if re.fullmatch(r"[0-9๐-๙]+", answer)
    ]
    if simple_answers:
        return Counter(simple_answers).most_common(1)[0][0]
    if answers:
        return Counter(answers).most_common(1)[0][0]
    return "2"
