from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from chest_disease.config import EXPECTED_COLUMNS, LABEL_COLUMNS, ProjectPaths, RunConfig
from chest_disease.pipeline import create_submission_from_probabilities
from scripts.build_colab_notebook import build_notebook


def test_create_submission_from_probabilities_writes_valid_binary_csv(tmp_path: Path):
    test_template = pd.DataFrame(
        [
            ["cxr00001.jpg"] + [""] * len(LABEL_COLUMNS),
            ["cxr00002.jpg"] + [""] * len(LABEL_COLUMNS),
        ],
        columns=EXPECTED_COLUMNS,
    )
    template_path = tmp_path / "test_submission.csv"
    test_template.to_csv(template_path, index=False)
    paths = ProjectPaths(
        competition_dir=tmp_path,
        image_dir=tmp_path / "images",
        train_csv=tmp_path / "train.csv",
        test_submission_csv=template_path,
        working_dir=tmp_path / "working",
        submission_path=tmp_path / "submission.csv",
    )
    probabilities = np.zeros((2, len(LABEL_COLUMNS)), dtype=float)
    probabilities[0, LABEL_COLUMNS.index("Atelectasis")] = 0.8
    thresholds = {label: 0.5 for label in LABEL_COLUMNS}
    result = create_submission_from_probabilities(paths, probabilities, thresholds, RunConfig(output_mode="binary"))
    submission = pd.read_csv(paths.submission_path)
    assert result.rows == 2
    assert submission.loc[0, "Atelectasis"] == 1
    assert submission.loc[0, "No Finding"] == 0
    assert submission.loc[1, "No Finding"] == 1


def test_build_colab_notebook_contains_required_cells(tmp_path: Path):
    output = tmp_path / "solution.ipynb"
    build_notebook(output)
    notebook = nbformat.read(output, as_version=4)
    joined = "\n".join(cell.get("source", "") for cell in notebook.cells)
    assert "pip install -q kaggle timm torchxrayvision iterative-stratification" in joined
    assert "Path('/content/Chest-Disease/src/chest_disease/config.py')" in joined
    assert "write_text" in joined
    assert "kaggle competitions download -c chest-disease-detection" in joined
    assert "/content/submission.csv" in joined
    assert "kaggle competitions submit" not in joined
