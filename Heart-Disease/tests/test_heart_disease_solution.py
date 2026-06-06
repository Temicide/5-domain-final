from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import heart_disease_solution as hds


def _base_rows(n: int = 6, include_target: bool = True) -> pd.DataFrame:
    rows = []
    for idx in range(n):
        row = {
            "ID": f"id_{idx}",
            "High Blood Pressure": "Yes" if idx % 2 else "No",
            "Told High Cholesterol": "No",
            "Cholesterol Checked": "Yes",
            "Body Mass Index": 22 + idx,
            "Smoked 100+ Cigarettes": "No",
            "Diagnosed Stroke": "No",
            "Diagnosed Diabetes": "Yes" if idx % 3 == 0 else "No",
            "Leisure Physical Activity": "Yes",
            "Heavy Alcohol Consumption": "No",
            "Health Care Coverage": "Yes",
            "Doctor Visit Cost Barrier": "No",
            "General Health": ["Excellent", "Very Good", "Good", "Fair", "Poor", "Good"][idx % 6],
            "Difficulty Walking": "Yes" if idx % 2 else "No",
            "Sex": "Female" if idx % 2 else "Male",
            "Education Level": "College",
            "Income Level": "$75,000 or more",
            "Age": 35 + idx * 5,
            "Vegetable or Fruit Intake (1+ per Day)": "Yes",
        }
        if include_target:
            row[hds.TARGET_COLUMN] = "Yes" if idx % 2 else "No"
        rows.append(row)
    columns = hds.EXPECTED_TRAIN_COLUMNS if include_target else hds.EXPECTED_TEST_COLUMNS
    return pd.DataFrame(rows)[columns]


def _sample_submission(test: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({hds.ID_COLUMN: test[hds.ID_COLUMN], hds.TARGET_COLUMN: hds.NEGATIVE_LABEL})


def test_column_and_path_contract_matches_spec():
    assert hds.ID_COLUMN == "ID"
    assert hds.TARGET_COLUMN == "History of HeartDisease or Attack"
    assert hds.SUBMISSION_COLUMNS == ["ID", "History of HeartDisease or Attack"]
    assert hds.EXPECTED_TEST_COLUMNS == [column for column in hds.EXPECTED_TRAIN_COLUMNS if column != hds.TARGET_COLUMN]
    assert hds.COMPETITION_SLUG == "super-ai-engineer-ss-6-individual-heart-disease-prediction"
    assert hds.COLAB_INPUT_ROOT == "/content/input"
    assert hds.COLAB_COMPETITION_DIR.endswith(hds.COMPETITION_SLUG)
    assert hds.COLAB_WORKING_DIR == "/content/working"
    assert hds.COLAB_SUBMISSION_PATH == "/content/submission.csv"
    assert hds.LOCAL_DATA_DIR.endswith(f"Heart-Disease/data/{hds.COMPETITION_SLUG}")
    assert hds.LOCAL_SUBMISSION_PATH.endswith("Heart-Disease/outputs/submissions/submission.csv")


def test_configure_kaggle_credentials_uses_environment_without_printing(capsys, tmp_path):
    env = {"KAGGLE_USERNAME": "secret-user", "KAGGLE_KEY": "secret-key"}
    status = hds.configure_kaggle_credentials(kaggle_dir=tmp_path, env=env)
    captured = capsys.readouterr()
    assert status == "environment_variables"
    assert "secret-user" not in captured.out
    assert "secret-key" not in captured.out
    assert not (tmp_path / "kaggle.json").exists()


def test_configure_kaggle_credentials_writes_uploaded_json_securely(capsys, tmp_path):
    payload = json.dumps({"username": "secret-user", "key": "secret-key"}).encode("utf-8")
    status = hds.configure_kaggle_credentials(kaggle_json_bytes=payload, kaggle_dir=tmp_path, env={})
    credential_path = tmp_path / "kaggle.json"
    captured = capsys.readouterr()
    assert status == "uploaded_kaggle_json"
    assert credential_path.read_bytes() == payload
    assert stat.S_IMODE(credential_path.stat().st_mode) == 0o600
    assert "secret-user" not in captured.out
    assert "secret-key" not in captured.out


def test_download_command_uses_kaggle_competitions_download(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, check):
        calls.append((command, check))
        (tmp_path / f"{hds.COMPETITION_SLUG}.zip").write_bytes(b"zip")

    monkeypatch.setattr(subprocess, "run", fake_run)
    archive = hds.download_competition_archive(input_root=tmp_path)
    assert archive == tmp_path / f"{hds.COMPETITION_SLUG}.zip"
    assert calls == [(["kaggle", "competitions", "download", "-c", hds.COMPETITION_SLUG, "-p", str(tmp_path)], True)]


def test_extract_competition_archive_creates_colab_input_dir(tmp_path):
    archive_path = tmp_path / "competition.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("train.csv", "ID,History of HeartDisease or Attack\n1,No\n")
        zf.writestr("test.csv", "ID\n2\n")
        zf.writestr("sample_submission.csv", "ID,History of HeartDisease or Attack\n2,No\n")
    target = tmp_path / hds.COMPETITION_SLUG
    assert hds.extract_competition_archive(archive_path, target) == target
    assert (target / "train.csv").exists()
    assert (target / "test.csv").exists()
    assert (target / "sample_submission.csv").exists()


def test_resolve_data_paths_prefers_extracted_colab_data(monkeypatch, tmp_path):
    colab_dir = tmp_path / "colab"
    colab_dir.mkdir()
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        (colab_dir / name).write_text("", encoding="utf-8")
    monkeypatch.setattr(hds, "COLAB_COMPETITION_DIR", str(colab_dir))
    paths = hds.resolve_data_paths()
    assert paths.train_path == colab_dir / "train.csv"
    assert paths.submission_path == Path(hds.COLAB_SUBMISSION_PATH)


def test_resolve_data_paths_falls_back_to_local_data(monkeypatch, tmp_path):
    monkeypatch.setattr(hds, "COLAB_COMPETITION_DIR", str(tmp_path / "missing"))
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        (local_dir / name).write_text("", encoding="utf-8")
    monkeypatch.setattr(hds, "LOCAL_DATA_DIR", str(local_dir))
    paths = hds.resolve_data_paths()
    assert paths.train_path == local_dir / "train.csv"
    assert paths.working_submission_path is None


def test_load_competition_data_reads_bom_and_preserves_id(tmp_path):
    train = _base_rows()
    test = _base_rows(include_target=False)
    sample = _sample_submission(test)
    train_path, test_path, sample_path = tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "sample_submission.csv"
    train_path.write_text("\ufeff" + train.to_csv(index=False), encoding="utf-8")
    test.to_csv(test_path, index=False, encoding="utf-8-sig")
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")
    data = hds.load_competition_data(hds.DataPaths(train_path, test_path, sample_path, tmp_path / "submission.csv", None))
    assert list(data.train.columns) == hds.EXPECTED_TRAIN_COLUMNS
    assert data.train[hds.ID_COLUMN].tolist() == train[hds.ID_COLUMN].tolist()


def test_validate_input_frames_rejects_missing_column():
    train = _base_rows().drop(columns=["Age"])
    test = _base_rows(include_target=False)
    with pytest.raises(ValueError, match="Train columns"):
        hds.validate_input_frames(train, test, _sample_submission(test))


def test_prepare_supervised_training_frame_drops_only_missing_target():
    train = _base_rows(4)
    train.loc[0, hds.TARGET_COLUMN] = np.nan
    train.loc[1, "Body Mass Index"] = np.nan
    supervised = hds.prepare_supervised_training_frame(train)
    assert len(supervised) == 3
    assert "id_1" in supervised[hds.ID_COLUMN].tolist()


def test_build_features_adds_auditable_health_risk_features():
    features = hds.build_features(_base_rows(include_target=False))
    for column in [
        "Body Mass Index",
        "Age",
        "age_bin",
        "bmi_class",
        "clinical_risk_count",
        "cardiometabolic_cluster",
        "lifestyle_protective_count",
        "health_access_friction",
        "education_rank",
        "income_rank",
        "age_x_high_bp",
        "general_health_x_walking",
    ]:
        assert column in features.columns
    assert pd.api.types.is_numeric_dtype(features["Body Mass Index"])
    assert pd.api.types.is_numeric_dtype(features["Age"])


def test_tune_threshold_prefers_f1_threshold_from_oof_predictions():
    result = hds.tune_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.9], metric="f1", thresholds=[0.2, 0.5])
    assert result["threshold"] == 0.2
    assert result["f1"] > 0.7


def test_validate_submission_rejects_blank_labels():
    test = _base_rows(2, include_target=False)
    submission = _sample_submission(test)
    submission.loc[0, hds.TARGET_COLUMN] = ""
    with pytest.raises(ValueError, match="blank"):
        hds.validate_submission(submission, test)


def test_write_submission_creates_valid_csv(tmp_path):
    test = _base_rows(3, include_target=False)
    submission = hds.make_submission(test, [0.1, 0.9, 0.3], threshold=0.5)
    path = hds.write_submission(submission, test, tmp_path / "submission.csv")
    loaded = pd.read_csv(path, encoding="utf-8-sig")
    assert loaded[hds.TARGET_COLUMN].tolist() == ["No", "Yes", "No"]


def test_train_logistic_smoke_model_returns_oof_and_test_probabilities():
    train = _base_rows(12)
    test = _base_rows(5, include_target=False)
    result = hds.train_single_model_cv(train, test, model_name="logistic", n_splits=3)
    assert len(result.oof_probabilities) == len(train)
    assert len(result.test_probabilities) == len(test)
    assert np.all((result.oof_probabilities >= 0) & (result.oof_probabilities <= 1))
    assert np.all((result.test_probabilities >= 0) & (result.test_probabilities <= 1))
    assert "roc_auc" in result.metrics
    assert "pr_auc" in result.metrics
    assert "f1" in result.threshold_metrics


def test_rank_average_predictions_combines_weighted_normalized_ranks():
    combined = hds.rank_average_predictions([[0.1, 0.8, 0.4], [0.3, 0.2, 0.9]], weights=[0.75, 0.25])
    assert combined.shape == (3,)
    assert np.all((combined >= 0) & (combined <= 1))
    assert combined[1] > combined[0]


def test_notebook_builder_contract():
    builder = Path(__file__).resolve().parents[1] / "scripts" / "build_colab_notebook.py"
    assert builder.exists()
    source = builder.read_text(encoding="utf-8")
    assert "heart_disease_colab_solution.ipynb" in source
    assert "configure_kaggle_credentials" in source
    assert "ensure_competition_data_available" in source
    assert "run_experiment" in source
    assert "/content/input" in source
    assert "/content/submission.csv" in source
    assert "kaggle competitions submit" not in source
    assert "competition_submit" not in source
