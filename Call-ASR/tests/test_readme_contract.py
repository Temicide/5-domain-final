from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_manual_submission_boundary_and_commands():
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "/content/submission.csv" in text
    assert "/content/input/individual-test-thai-call-center-asr" in text
    assert "kaggle.json" in text
    assert "KAGGLE_USERNAME" in text
    assert "KAGGLE_KEY" in text
    assert "python3 -m call_asr.audit_audio" in text
    assert "python3 -m call_asr.infer" in text
    assert "python3 -m pytest -v" in text
    assert "Do not run kaggle competitions submit" in text
