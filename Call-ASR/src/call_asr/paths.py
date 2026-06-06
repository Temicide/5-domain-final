from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


COMPETITION_SLUG = "individual-test-thai-call-center-asr"


@dataclass(frozen=True)
class CompetitionPaths:
    input_dir: Path
    audio_dir: Path
    sample_submission: Path
    working_dir: Path
    submissions_dir: Path
    is_colab: bool


def _has_competition_files(input_dir: Path) -> bool:
    return (input_dir / "sample_submission.csv").is_file() and (input_dir / "audio_final" / "audio").is_dir()


def resolve_competition_paths(
    colab_input_root: Path = Path("/content/input"),
    project_root: Path | None = None,
) -> CompetitionPaths:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    colab_competition_dir = colab_input_root / COMPETITION_SLUG
    local_competition_dir = project_root / "data" / COMPETITION_SLUG

    if _has_competition_files(colab_competition_dir):
        input_dir = colab_competition_dir
        working_dir = Path("/content/working")
        submissions_dir = Path("/content/working")
        is_colab = True
    elif _has_competition_files(local_competition_dir):
        input_dir = local_competition_dir
        working_dir = project_root / "data" / "runs"
        submissions_dir = project_root / "data" / "submissions"
        is_colab = False
    else:
        raise FileNotFoundError(
            "Competition files not found. Expected sample_submission.csv and "
            f"audio_final/audio under {colab_competition_dir} or {local_competition_dir}."
        )

    return CompetitionPaths(
        input_dir=input_dir,
        audio_dir=input_dir / "audio_final" / "audio",
        sample_submission=input_dir / "sample_submission.csv",
        working_dir=working_dir,
        submissions_dir=submissions_dir,
        is_colab=is_colab,
    )
