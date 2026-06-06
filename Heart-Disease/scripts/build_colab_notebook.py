from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "heart_disease_solution.py"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "heart_disease_colab_solution.ipynb"


def main() -> None:
    module_source = MODULE_PATH.read_text(encoding="utf-8")
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "colab": {"name": "heart_disease_colab_solution.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Heart Disease Kaggle Submission\n\n"
            "Official metric checked in Kaggle UI on 2026-06-06: use the displayed competition metric to choose `threshold_metric`.\n\n"
            "This notebook downloads the competition CSVs into `/content/input`, trains tabular models with OOF threshold tuning, "
            "and writes `/content/submission.csv`."
        ),
        nbf.v4.new_code_cell("!pip -q install kaggle lightgbm catboost xgboost"),
        nbf.v4.new_code_cell(
            "import os\n"
            "from pathlib import Path\n\n"
            "credential_status = None\n"
            "if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):\n"
            "    credential_status = 'environment_variables'\n"
            "else:\n"
            "    from google.colab import files\n"
            "    uploaded = files.upload()\n"
            "    kaggle_json = uploaded.get('kaggle.json')\n"
            "    if kaggle_json is None:\n"
            "        raise RuntimeError('Upload kaggle.json or set KAGGLE_USERNAME and KAGGLE_KEY.')\n"
            "print('credential_source_ready')"
        ),
        nbf.v4.new_code_cell(module_source),
        nbf.v4.new_code_cell(
            "if credential_status == 'environment_variables':\n"
            "    configure_kaggle_credentials()\n"
            "else:\n"
            "    configure_kaggle_credentials(kaggle_json_bytes=kaggle_json)\n"
            "data_dir = ensure_competition_data_available(force_download=False)\n"
            "print('data_dir', data_dir)"
        ),
        nbf.v4.new_code_cell(
            "summary = run_experiment(\n"
            "    model_names=['lightgbm', 'catboost', 'xgboost', 'extratrees', 'logistic'],\n"
            "    threshold_metric='f1',\n"
            ")\n"
            "print('submission_rows', summary['submission_rows'])\n"
            "print('positive_predictions', summary['positive_predictions'])\n"
            "print('ensemble_metrics', summary['ensemble_metrics'])\n"
            "print('threshold_metrics', summary['threshold_metrics'])\n"
            "print('submission_path', summary['submission_path'])"
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "submission = pd.read_csv('/content/submission.csv', encoding='utf-8-sig')\n"
            "test = pd.read_csv('/content/input/super-ai-engineer-ss-6-individual-heart-disease-prediction/test.csv', encoding='utf-8-sig')\n"
            "assert list(submission.columns) == ['ID', 'History of HeartDisease or Attack']\n"
            "assert len(submission) == 74361\n"
            "assert submission['ID'].tolist() == test['ID'].tolist()\n"
            "assert set(submission['History of HeartDisease or Attack']) <= {'Yes', 'No'}\n"
            "assert submission['History of HeartDisease or Attack'].isna().sum() == 0\n"
            "print(submission.shape)\n"
            "print(submission['History of HeartDisease or Attack'].value_counts().to_dict())"
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
