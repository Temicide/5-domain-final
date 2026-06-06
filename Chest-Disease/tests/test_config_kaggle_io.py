from pathlib import Path

from chest_disease import __version__
from chest_disease.config import (
    COLAB_ALT_COMPETITION_DIR,
    COLAB_COMPETITION_DIR,
    COLAB_INPUT_ROOT,
    COLAB_PACKAGE_ROOT,
    COLAB_PACKAGE_SRC,
    COLAB_SUBMISSION_PATH,
    COLAB_WORKING_DIR,
    COMPETITION_SLUG,
    DISEASE_LABEL_COLUMNS,
    EXPECTED_COLUMNS,
    ID_COLUMN,
    LABEL_COLUMNS,
    LOCAL_COMPETITION_DIRNAME,
    LOCAL_DATA_DIR,
    LOCAL_IMAGE_DIR,
    LOCAL_SUBMISSION_PATH,
    PROJECT_ROOT,
    RunConfig,
)

import zipfile

import pytest

from chest_disease.kaggle_io import (
    build_download_command,
    configure_kaggle_credentials,
    extract_archive,
    resolve_project_paths,
)


def test_constants_match_chest_disease_spec():
    assert __version__ == "0.1.0"
    assert COMPETITION_SLUG == "chest-disease-detection"
    assert LOCAL_COMPETITION_DIRNAME == "individual-test-chest-disease-detection"
    assert ID_COLUMN == "filename"
    assert LABEL_COLUMNS == [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
    ]
    assert DISEASE_LABEL_COLUMNS == [label for label in LABEL_COLUMNS if label != "No Finding"]
    assert EXPECTED_COLUMNS == [ID_COLUMN] + LABEL_COLUMNS
    assert PROJECT_ROOT == Path("/Users/temicide/Documents/5_domain_final/Chest-Disease")
    assert COLAB_INPUT_ROOT == Path("/content/input")
    assert COLAB_WORKING_DIR == Path("/content/working")
    assert COLAB_COMPETITION_DIR == Path("/content/input/chest-disease-detection")
    assert COLAB_ALT_COMPETITION_DIR == Path("/content/input/individual-test-chest-disease-detection")
    assert COLAB_SUBMISSION_PATH == Path("/content/submission.csv")
    assert COLAB_PACKAGE_ROOT == Path("/content/Chest-Disease")
    assert COLAB_PACKAGE_SRC == Path("/content/Chest-Disease/src")
    assert LOCAL_DATA_DIR == PROJECT_ROOT / "data" / LOCAL_COMPETITION_DIRNAME
    assert LOCAL_IMAGE_DIR == LOCAL_DATA_DIR / "images" / "images"
    assert LOCAL_SUBMISSION_PATH == PROJECT_ROOT / "outputs" / "submissions" / "submission.csv"


def test_run_config_defaults_are_a100_friendly():
    config = RunConfig()
    assert config.image_size == 512
    assert config.batch_size == 32
    assert config.num_folds == 5
    assert config.seed == 42
    assert config.allow_external_weights is True
    assert config.use_amp is True
    assert config.output_mode == "binary"


def test_build_download_command_uses_competition_slug():
    command = build_download_command(input_root=Path("/content/input"))
    assert command == [
        "kaggle",
        "competitions",
        "download",
        "-c",
        "chest-disease-detection",
        "-p",
        "/content/input",
    ]


def test_configure_kaggle_credentials_uses_env_without_printing(capsys):
    env = {"KAGGLE_USERNAME": "user-name", "KAGGLE_KEY": "secret-key"}
    status = configure_kaggle_credentials(env=env, kaggle_json_bytes=None, kaggle_dir=Path("/tmp/not-used"))
    captured = capsys.readouterr()
    assert status == "environment_variables"
    assert "secret-key" not in captured.out
    assert "user-name" not in captured.out


def test_configure_kaggle_credentials_writes_uploaded_json_securely(tmp_path: Path, capsys):
    kaggle_dir = tmp_path / ".kaggle"
    payload = b'{"username":"uploaded-user","key":"uploaded-secret"}'
    status = configure_kaggle_credentials(env={}, kaggle_json_bytes=payload, kaggle_dir=kaggle_dir)
    target = kaggle_dir / "kaggle.json"
    captured = capsys.readouterr()
    assert status == "uploaded_kaggle_json"
    assert target.read_bytes() == payload
    assert oct(target.stat().st_mode & 0o777) == "0o600"
    assert "uploaded-secret" not in captured.out


def test_configure_kaggle_credentials_raises_when_missing(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Kaggle credentials are missing"):
        configure_kaggle_credentials(env={}, kaggle_json_bytes=None, kaggle_dir=tmp_path)


def test_extract_archive_and_resolve_paths_prefers_colab(tmp_path: Path):
    input_root = tmp_path / "input"
    archive = input_root / "chest-disease-detection.zip"
    image_file = "chest-disease-detection/images/images/cxr00001.jpg"
    input_root.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("chest-disease-detection/train.csv", "filename,Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,Pneumothorax\n")
        zf.writestr("chest-disease-detection/test_submission.csv", "filename,Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,Pneumothorax\n")
        zf.writestr(image_file, b"fake")
    extracted = extract_archive(archive, input_root / "chest-disease-detection")
    paths = resolve_project_paths(
        colab_input_root=input_root,
        local_data_dir=tmp_path / "local-missing",
        working_dir=tmp_path / "working",
        submission_path=tmp_path / "submission.csv",
    )
    assert extracted == input_root / "chest-disease-detection"
    assert paths.competition_dir == input_root / "chest-disease-detection"
    assert paths.train_csv.name == "train.csv"
    assert paths.test_submission_csv.name == "test_submission.csv"
    assert paths.image_dir.name == "images"
