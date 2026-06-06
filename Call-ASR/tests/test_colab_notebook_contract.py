from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SCRIPT = ROOT / "notebooks" / "colab_submission.py"
NOTEBOOK = ROOT / "notebooks" / "colab_submission.ipynb"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_colab_notebook_file_exists_and_is_uploadable():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 5
    assert any(cell["cell_type"] == "markdown" for cell in notebook["cells"])
    assert any(cell["cell_type"] == "code" for cell in notebook["cells"])


def test_notebook_script_writes_required_submission_path():
    source = NOTEBOOK_SCRIPT.read_text(encoding="utf-8") + _notebook_source()

    assert "/content/submission.csv" in source
    assert "/content/input/individual-test-thai-call-center-asr" in source
    assert "/content/working" in source
    assert "write_submission_csv" in source
    assert "validate_submission_frame" in source


def test_notebook_script_downloads_and_extracts_before_reading():
    source = NOTEBOOK_SCRIPT.read_text(encoding="utf-8") + _notebook_source()

    assert "ensure_kaggle_package" in source
    assert "configure_kaggle_credentials" in source
    assert "download_and_extract_competition_data" in source
    assert '"competitions"' in source
    assert '"download"' in source
    assert ".extractall" in source
    assert source.index("download_and_extract_competition_data()") < source.index("resolve_competition_paths(")


def test_notebook_script_does_not_print_credentials():
    source = (NOTEBOOK_SCRIPT.read_text(encoding="utf-8") + _notebook_source()).lower()
    assert "print(kaggle_username" not in source
    assert "print(kaggle_key" not in source
    assert "print(credentials" not in source


def test_notebook_script_does_not_submit_to_kaggle():
    source = (NOTEBOOK_SCRIPT.read_text(encoding="utf-8") + _notebook_source()).lower()
    forbidden = [
        "kaggle competitions submit",
        ".competitions.submit",
        "api.competition_submit",
        "competition_submit(",
    ]

    assert [needle for needle in forbidden if needle in source] == []
