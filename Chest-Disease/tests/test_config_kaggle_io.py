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
