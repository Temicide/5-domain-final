from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path("/Users/temicide/Documents/5_domain_final/Chest-Disease")
COLAB_PROJECT_ROOT = Path("/content/Chest-Disease")
SOURCE_FILES = [
    "src/chest_disease/__init__.py",
    "src/chest_disease/config.py",
    "src/chest_disease/kaggle_io.py",
    "src/chest_disease/data.py",
    "src/chest_disease/folds.py",
    "src/chest_disease/metrics.py",
    "src/chest_disease/dataset.py",
    "src/chest_disease/models.py",
    "src/chest_disease/train.py",
    "src/chest_disease/pipeline.py",
]


def source_file_cells() -> list:
    cells = []
    for relative in SOURCE_FILES:
        source = (PROJECT_ROOT / relative).read_text()
        colab_path = COLAB_PROJECT_ROOT / relative
        cells.append(
            nbf.v4.new_code_cell(
                "from pathlib import Path\n"
                f"path = Path({str(colab_path)!r})\n"
                "path.parent.mkdir(parents=True, exist_ok=True)\n"
                f"path.write_text({source!r})\n"
            )
        )
    return cells


def build_notebook(output_path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Chest Disease Detection Colab A100 Solution"),
        nbf.v4.new_code_cell("!pip install -q kaggle timm torchxrayvision iterative-stratification nbformat"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import os, zipfile, subprocess\n"
            "INPUT_ROOT = Path('/content/input')\n"
            "INPUT_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "WORKING = Path('/content/working')\n"
            "WORKING.mkdir(parents=True, exist_ok=True)\n"
        ),
        nbf.v4.new_code_cell(
            "from google.colab import files\n"
            "try:\n"
            "    from google.colab import userdata\n"
            "    if not os.environ.get('KAGGLE_USERNAME'):\n"
            "        os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME') or ''\n"
            "    if not os.environ.get('KAGGLE_KEY'):\n"
            "        os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY') or ''\n"
            "except Exception:\n"
            "    pass\n"
            "kaggle_json = Path('/root/.kaggle/kaggle.json')\n"
            "if not (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')) and not kaggle_json.exists():\n"
            "    uploaded = files.upload()\n"
            "    if 'kaggle.json' not in uploaded:\n"
            "        raise RuntimeError('Upload kaggle.json or set KAGGLE_USERNAME/KAGGLE_KEY')\n"
            "    kaggle_json.parent.mkdir(parents=True, exist_ok=True)\n"
            "    kaggle_json.write_bytes(uploaded['kaggle.json'])\n"
            "    kaggle_json.chmod(0o600)\n"
        ),
        nbf.v4.new_code_cell(
            "!kaggle competitions download -c chest-disease-detection -p /content/input\n"
            "archive = Path('/content/input/chest-disease-detection.zip')\n"
            "with zipfile.ZipFile(archive) as zf:\n"
            "    zf.extractall('/content/input')\n"
        ),
        *source_file_cells(),
        nbf.v4.new_code_cell(
            "import sys\n"
            "sys.path.insert(0, '/content/Chest-Disease/src')\n"
            "from chest_disease.config import RunConfig\n"
            "from chest_disease.kaggle_io import resolve_project_paths\n"
            "from chest_disease.pipeline import run_colab_pipeline\n"
            "paths = resolve_project_paths()\n"
            "result = run_colab_pipeline(paths, RunConfig())\n"
            "print(f'Wrote /content/submission.csv with {result.rows} rows')\n"
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output_path)


if __name__ == "__main__":
    build_notebook(
        Path("/Users/temicide/Documents/5_domain_final/Chest-Disease/notebooks/chest_disease_colab_a100_solution.ipynb")
    )
