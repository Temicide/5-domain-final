from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABEL_COLUMNS, ProjectPaths, RunConfig
from .data import validate_submission
from .metrics import apply_no_finding_consistency


def create_submission_from_probabilities(
    paths: ProjectPaths,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
    config: RunConfig,
):
    template = pd.read_csv(paths.test_submission_csv)
    if probabilities.shape != (len(template), len(LABEL_COLUMNS)):
        raise ValueError(f"probability shape {probabilities.shape} does not match test rows and labels")
    output = template.copy()
    if config.output_mode == "probability":
        output[LABEL_COLUMNS] = probabilities
    elif config.output_mode == "binary":
        binary = pd.DataFrame(probabilities, columns=LABEL_COLUMNS)
        for label in LABEL_COLUMNS:
            binary[label] = (binary[label] >= thresholds[label]).astype(int)
        output[LABEL_COLUMNS] = apply_no_finding_consistency(binary).astype(int)
    else:
        raise ValueError("output_mode must be 'binary' or 'probability'")
    paths.submission_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(paths.submission_path, index=False)
    return validate_submission(paths.submission_path, paths.test_submission_csv, config.output_mode)


def ensure_working_dirs(paths: ProjectPaths) -> None:
    for name in ["oof", "test_preds", "thresholds", "submissions", "checkpoints", "logs"]:
        (paths.working_dir / name).mkdir(parents=True, exist_ok=True)


def save_probabilities(path: Path, filenames: list[str], probabilities: np.ndarray) -> None:
    frame = pd.DataFrame(probabilities, columns=LABEL_COLUMNS)
    frame.insert(0, "filename", filenames)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_probability_csv(path: Path) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(path)
    return frame["filename"].astype(str).tolist(), frame[LABEL_COLUMNS].astype(float).to_numpy()


def run_colab_pipeline(paths: ProjectPaths, config: RunConfig):
    ensure_working_dirs(paths)
    test_template = pd.read_csv(paths.test_submission_csv)
    prevalence = np.zeros((len(test_template), len(LABEL_COLUMNS)), dtype=float)
    prevalence[:, LABEL_COLUMNS.index("No Finding")] = 1.0
    thresholds = {label: 0.5 for label in LABEL_COLUMNS}
    save_probabilities(
        paths.working_dir / "test_preds" / "prevalence_baseline.csv",
        test_template["filename"].astype(str).tolist(),
        prevalence,
    )
    return create_submission_from_probabilities(paths, prevalence, thresholds, config)
