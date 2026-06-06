# %% [markdown]
# # Thai Call Center ASR Submission
# This notebook downloads Kaggle competition data, generates /content/submission.csv,
# and stops before any upload or submission step.

# %%
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import zipfile

COMPETITION_SLUG = "individual-test-thai-call-center-asr"
CONTENT_ROOT = Path("/content")
INPUT_DIR = Path("/content/input/individual-test-thai-call-center-asr")
WORKING_DIR = Path("/content/working")
SUBMISSION_PATH = CONTENT_ROOT / "submission.csv"
PROJECT_ROOT = CONTENT_ROOT / "Call-ASR"
LOCAL_PROJECT_ROOT = Path("/Users/temicide/Documents/5_domain_final/Call-ASR")
if LOCAL_PROJECT_ROOT.exists():
    PROJECT_ROOT = LOCAL_PROJECT_ROOT
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def ensure_kaggle_package() -> None:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "kaggle"],
            check=True,
        )


def configure_kaggle_credentials() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json_path = kaggle_dir / "kaggle.json"

    uploaded_json = Path("/content/kaggle.json")
    if uploaded_json.exists():
        shutil.copyfile(uploaded_json, kaggle_json_path)
    else:
        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")
        if not username or not key:
            try:
                from google.colab import userdata

                username = username or userdata.get("KAGGLE_USERNAME")
                key = key or userdata.get("KAGGLE_KEY")
            except Exception:
                pass
        if not username or not key:
            raise RuntimeError(
                "Provide Kaggle credentials via uploaded /content/kaggle.json "
                "or Colab secrets/environment variables KAGGLE_USERNAME and KAGGLE_KEY."
            )
        kaggle_json_path.write_text(
            json.dumps({"username": username, "key": key}),
            encoding="utf-8",
        )
    kaggle_json_path.chmod(0o600)


def download_and_extract_competition_data() -> Path:
    ensure_kaggle_package()
    configure_kaggle_credentials()
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_submission = INPUT_DIR / "sample_submission.csv"
    audio_dir = INPUT_DIR / "audio_final" / "audio"
    if sample_submission.exists() and audio_dir.is_dir():
        return INPUT_DIR

    archive_path = WORKING_DIR / f"{COMPETITION_SLUG}.zip"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION_SLUG,
            "-p",
            str(WORKING_DIR),
            "--force",
        ],
        check=True,
    )
    if not archive_path.exists():
        raise FileNotFoundError(f"Downloaded archive not found: {archive_path}")
    with zipfile.ZipFile(archive_path) as zip_file:
        zip_file.extractall(INPUT_DIR)
    return INPUT_DIR


# %%
from call_asr.infer import WhisperPipelineBackend, run_inference
from call_asr.paths import resolve_competition_paths
from call_asr.submission import (
    load_sample_submission,
    validate_audio_coverage,
    validate_submission_frame,
    write_submission_csv,
)

# %%
MODEL_NAME = "typhoon-ai/typhoon-whisper-large-v3"
NORMALIZATION_POLICY = "single_space"
CHUNK_LENGTH_SECONDS = 30
BATCH_SIZE = 4
CONDITION_ON_PREVIOUS_TEXT = False

# %%
download_and_extract_competition_data()
paths = resolve_competition_paths(colab_input_root=CONTENT_ROOT / "input", project_root=PROJECT_ROOT)
sample_df = load_sample_submission(paths.sample_submission)
validate_audio_coverage(sample_df, paths.audio_dir)
print(f"Rows in sample submission: {len(sample_df)}")
print(f"Audio directory: {paths.audio_dir}")
print(f"Output submission path: {SUBMISSION_PATH}")

# %%
backend = WhisperPipelineBackend(
    model_name=MODEL_NAME,
    chunk_length_s=CHUNK_LENGTH_SECONDS,
    batch_size=BATCH_SIZE,
    condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
    device="cuda",
)

# %%
safe_model_name = MODEL_NAME.replace("/", "__")
predictions_path = paths.working_dir / f"predictions_{safe_model_name}_{NORMALIZATION_POLICY}.csv"
log_path = paths.working_dir / f"run_{safe_model_name}_{NORMALIZATION_POLICY}.jsonl"
predictions_df = run_inference(
    sample_df=sample_df,
    audio_dir=paths.audio_dir,
    output_csv=predictions_path,
    log_jsonl=log_path,
    backend=backend,
    normalization_policy=NORMALIZATION_POLICY,
    resume=True,
)
print(f"Wrote predictions: {predictions_path}")
print(f"Wrote logs: {log_path}")
print(f"Prediction rows: {len(predictions_df)}")
print(f"Inference errors: {(predictions_df['error'] != '').sum()}")

# %%
submission_df = write_submission_csv(
    sample_df=sample_df,
    predictions_df=predictions_df,
    output_path=SUBMISSION_PATH,
    allow_empty_files=set(),
)
validate_submission_frame(submission_df, allow_empty_files=set())
print(f"Wrote validated submission: {SUBMISSION_PATH}")
print(submission_df.head().to_string(index=False))
