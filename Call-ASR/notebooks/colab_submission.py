# %% [markdown]
# # Thai Call Center ASR Submission
# This notebook downloads Kaggle competition data, generates /content/submission.csv,
# and stops before any upload or submission step.

# %%
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

COLAB_REQUIRED_PACKAGES = [
    "kaggle",
    "pandas>=2.2.2",
    "numpy>=1.26.4",
    "soundfile>=0.12.1",
    "librosa>=0.10.2.post1",
    "jiwer>=3.0.4",
    "accelerate>=0.31.0",
    "transformers>=4.41.2",
]
COMPETITION_SLUG = "individual-test-thai-call-center-asr"
CONTENT_ROOT = Path("/content")
INPUT_DIR = Path("/content/input/individual-test-thai-call-center-asr")
WORKING_DIR = Path("/content/working")
SUBMISSION_PATH = CONTENT_ROOT / "submission.csv"
LOGGER = logging.getLogger("colab_submission")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    LOGGER.info("Logging initialized")
    LOGGER.info("Python: %s", sys.version.replace("\n", " "))
    LOGGER.info("Platform: %s", platform.platform())
    LOGGER.info("Working directory: %s", Path.cwd())
    LOGGER.info("Content root: %s", CONTENT_ROOT)
    LOGGER.info("Input dir: %s", INPUT_DIR)
    LOGGER.info("Working dir: %s", WORKING_DIR)
    LOGGER.info("Submission path: %s", SUBMISSION_PATH)


def _elapsed(start_time: float) -> str:
    return f"{time.perf_counter() - start_time:.1f}s"


def _run_logged(command: list[str], description: str) -> None:
    LOGGER.info("Starting: %s", description)
    LOGGER.info("Command: %s", " ".join(command))
    start_time = time.perf_counter()
    subprocess.run(command, check=True)
    LOGGER.info("Finished: %s in %s", description, _elapsed(start_time))


def install_colab_dependencies() -> None:
    LOGGER.info("Installing/upgrading %d Colab packages", len(COLAB_REQUIRED_PACKAGES))
    LOGGER.info("Packages: %s", ", ".join(COLAB_REQUIRED_PACKAGES))
    _run_logged(
        [sys.executable, "-m", "pip", "install", "--upgrade", *COLAB_REQUIRED_PACKAGES],
        "pip install Colab dependencies",
    )


def configure_kaggle_credentials() -> None:
    LOGGER.info("Configuring Kaggle credentials")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json_path = kaggle_dir / "kaggle.json"

    uploaded_json = Path("/content/kaggle.json")
    if uploaded_json.exists():
        LOGGER.info("Using uploaded Kaggle token at %s", uploaded_json)
        shutil.copyfile(uploaded_json, kaggle_json_path)
    else:
        LOGGER.info("Uploaded Kaggle token not found; checking environment variables and Colab secrets")
        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")
        LOGGER.info("KAGGLE_USERNAME present: %s", bool(username))
        LOGGER.info("KAGGLE_KEY present: %s", bool(key))
        if not username or not key:
            try:
                from google.colab import userdata

                username = username or userdata.get("KAGGLE_USERNAME")
                key = key or userdata.get("KAGGLE_KEY")
                LOGGER.info("Checked Colab userdata secrets for Kaggle credentials")
            except Exception:
                LOGGER.exception("Could not read Colab userdata secrets")
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
    LOGGER.info("Kaggle credential file ready at %s with mode 600", kaggle_json_path)


def download_and_extract_competition_data() -> Path:
    LOGGER.info("Preparing competition data for %s", COMPETITION_SLUG)
    configure_kaggle_credentials()
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Ensured directories exist: %s and %s", WORKING_DIR, INPUT_DIR)
    sample_submission = INPUT_DIR / "sample_submission.csv"
    audio_dir = INPUT_DIR / "audio_final" / "audio"
    if sample_submission.exists() and audio_dir.is_dir():
        audio_count = sum(1 for path in audio_dir.iterdir() if path.is_file())
        LOGGER.info("Competition data already present; sample=%s audio_files=%d", sample_submission, audio_count)
        return INPUT_DIR

    archive_path = WORKING_DIR / f"{COMPETITION_SLUG}.zip"
    _run_logged(
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
        "Kaggle competition download",
    )
    if not archive_path.exists():
        raise FileNotFoundError(f"Downloaded archive not found: {archive_path}")
    LOGGER.info("Downloaded archive: %s (%d bytes)", archive_path, archive_path.stat().st_size)
    start_time = time.perf_counter()
    with zipfile.ZipFile(archive_path) as zip_file:
        names = zip_file.namelist()
        LOGGER.info("Extracting %d archive members to %s", len(names), INPUT_DIR)
        zip_file.extractall(INPUT_DIR)
    LOGGER.info("Finished extracting archive in %s", _elapsed(start_time))
    return INPUT_DIR


# %%
setup_logging()
install_colab_dependencies()

import pandas as pd


class SubmissionValidationError(ValueError):
    pass


@dataclass(frozen=True)
class CompetitionPaths:
    input_dir: Path
    audio_dir: Path
    sample_submission: Path
    working_dir: Path
    submissions_dir: Path
    is_colab: bool


def _has_competition_files(input_dir: Path) -> bool:
    has_files = (input_dir / "sample_submission.csv").is_file() and (input_dir / "audio_final" / "audio").is_dir()
    LOGGER.info("Competition file check at %s: %s", input_dir, has_files)
    return has_files


def resolve_competition_paths(colab_input_root: Path = Path("/content/input")) -> CompetitionPaths:
    colab_competition_dir = colab_input_root / COMPETITION_SLUG
    LOGGER.info("Resolving competition paths under %s", colab_input_root)
    if not _has_competition_files(colab_competition_dir):
        raise FileNotFoundError(
            "Competition files not found. Expected sample_submission.csv and "
            f"audio_final/audio under {colab_competition_dir}. Run download_and_extract_competition_data() first."
        )
    paths = CompetitionPaths(
        input_dir=colab_competition_dir,
        audio_dir=colab_competition_dir / "audio_final" / "audio",
        sample_submission=colab_competition_dir / "sample_submission.csv",
        working_dir=WORKING_DIR,
        submissions_dir=WORKING_DIR,
        is_colab=True,
    )
    LOGGER.info("Resolved input_dir=%s", paths.input_dir)
    LOGGER.info("Resolved audio_dir=%s", paths.audio_dir)
    LOGGER.info("Resolved sample_submission=%s", paths.sample_submission)
    LOGGER.info("Resolved working_dir=%s", paths.working_dir)
    return paths


def load_sample_submission(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading sample submission from %s", path)
    df = pd.read_csv(path, dtype={"file_name": "string", "text": "string"}, keep_default_na=False)
    LOGGER.info("Loaded sample submission shape=%s columns=%s", df.shape, list(df.columns))
    if list(df.columns) != ["file_name", "text"]:
        raise SubmissionValidationError(
            f"Expected sample submission columns ['file_name', 'text'], got {list(df.columns)}"
        )
    if df["file_name"].duplicated().any():
        duplicates = df.loc[df["file_name"].duplicated(), "file_name"].tolist()
        raise SubmissionValidationError(f"Duplicate file_name values: {', '.join(duplicates[:10])}")
    LOGGER.info("Sample submission validated with %d unique file names", len(df))
    return df


def validate_audio_coverage(sample_df: pd.DataFrame, audio_dir: Path) -> None:
    LOGGER.info("Validating audio coverage in %s for %d rows", audio_dir, len(sample_df))
    missing = [name for name in sample_df["file_name"].tolist() if not (audio_dir / name).is_file()]
    if missing:
        LOGGER.error("Missing %d audio files; first missing values: %s", len(missing), missing[:20])
        raise SubmissionValidationError(f"Missing audio files: {', '.join(missing[:20])}")
    LOGGER.info("Audio coverage validation passed")


def validate_submission_frame(submission_df: pd.DataFrame, allow_empty_files: set[str] | None = None) -> None:
    allow_empty_files = allow_empty_files or set()
    LOGGER.info("Validating submission frame shape=%s allow_empty_files=%d", submission_df.shape, len(allow_empty_files))
    if list(submission_df.columns) != ["file_name", "text"]:
        raise SubmissionValidationError(
            f"Expected submission columns ['file_name', 'text'], got {list(submission_df.columns)}"
        )
    if submission_df["file_name"].duplicated().any():
        duplicates = submission_df.loc[submission_df["file_name"].duplicated(), "file_name"].tolist()
        raise SubmissionValidationError(f"Duplicate submission rows: {', '.join(duplicates[:10])}")
    for row in submission_df.itertuples(index=False):
        text = "" if pd.isna(row.text) else str(row.text)
        if text == "" and row.file_name not in allow_empty_files:
            raise SubmissionValidationError(f"Empty transcript for {row.file_name}")
    LOGGER.info("Submission frame validation passed")


def write_submission_csv(
    sample_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_path: Path,
    allow_empty_files: set[str] | None = None,
) -> pd.DataFrame:
    LOGGER.info("Writing submission CSV to %s", output_path)
    LOGGER.info("Sample rows=%d prediction rows=%d", len(sample_df), len(predictions_df))
    required_prediction_columns = {"file_name", "normalized_text"}
    missing_columns = required_prediction_columns - set(predictions_df.columns)
    if missing_columns:
        raise SubmissionValidationError(f"Prediction columns missing: {', '.join(sorted(missing_columns))}")

    merged = sample_df[["file_name"]].merge(
        predictions_df[["file_name", "normalized_text"]],
        on="file_name",
        how="left",
        validate="one_to_one",
    )
    if merged["normalized_text"].isna().any():
        missing = merged.loc[merged["normalized_text"].isna(), "file_name"].tolist()
        LOGGER.error("Missing %d predictions; first missing values: %s", len(missing), missing[:20])
        raise SubmissionValidationError(f"Missing predictions: {', '.join(missing[:20])}")

    submission_df = merged.rename(columns={"normalized_text": "text"})[["file_name", "text"]]
    submission_df["text"] = submission_df["text"].fillna("").astype(str)
    validate_submission_frame(submission_df, allow_empty_files=allow_empty_files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    LOGGER.info("Wrote submission CSV rows=%d bytes=%d", len(submission_df), output_path.stat().st_size)
    return submission_df


def normalize_text(text: str, policy: str) -> str:
    normalized = str(text).strip()
    if policy == "none":
        return normalized
    if policy == "single_space":
        return " ".join(normalized.split())
    raise ValueError(f"Unknown normalization policy: {policy}")


PREDICTION_COLUMNS = [
    "file_name",
    "raw_text",
    "normalized_text",
    "model_name",
    "avg_logprob",
    "compression_ratio",
    "no_speech_prob",
    "error",
]


@dataclass(frozen=True)
class AsrResult:
    text: str
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class WhisperPipelineBackend:
    def __init__(
        self,
        model_name: str,
        chunk_length_s: int,
        batch_size: int,
        condition_on_previous_text: bool,
        device: str,
    ):
        LOGGER.info("Initializing WhisperPipelineBackend")
        LOGGER.info("Model=%s chunk_length_s=%s batch_size=%s condition_on_previous_text=%s device=%s", model_name, chunk_length_s, batch_size, condition_on_previous_text, device)
        start_time = time.perf_counter()
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. In Colab, select Runtime > Change runtime type > A100 GPU.")
        self.model_name = model_name
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        device_index = 0 if device == "cuda" else -1
        LOGGER.info("Torch version=%s CUDA available=%s dtype=%s device_index=%s", torch.__version__, torch.cuda.is_available(), torch_dtype, device_index)
        if torch.cuda.is_available():
            LOGGER.info("CUDA device count=%d active_device=%s", torch.cuda.device_count(), torch.cuda.get_device_name(0))
        LOGGER.info("Loading speech seq2seq model from %s", model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        LOGGER.info("Loading processor from %s", model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        LOGGER.info("Building transformers ASR pipeline")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            device=device_index,
            return_timestamps=False,
        )
        self.generate_kwargs = {
            "language": "thai",
            "task": "transcribe",
            "condition_on_prev_tokens": condition_on_previous_text,
        }
        LOGGER.info("WhisperPipelineBackend ready in %s", _elapsed(start_time))

    def transcribe(self, audio_path: Path) -> AsrResult:
        LOGGER.info("Transcribing audio: %s", audio_path)
        start_time = time.perf_counter()
        output = self.pipe(str(audio_path), generate_kwargs=self.generate_kwargs)
        text = str(output.get("text", ""))
        LOGGER.info("Finished audio: %s chars=%d elapsed=%s", audio_path.name, len(text), _elapsed(start_time))
        return AsrResult(text=text, avg_logprob=0.0, compression_ratio=0.0, no_speech_prob=0.0)

    def transcribe_many(self, audio_paths: list[Path]) -> list[AsrResult]:
        if not audio_paths:
            return []
        LOGGER.info("Transcribing batch of %d audio files", len(audio_paths))
        start_time = time.perf_counter()
        outputs = self.pipe([str(path) for path in audio_paths], generate_kwargs=self.generate_kwargs)
        results = [AsrResult(text=str(output.get("text", "")), avg_logprob=0.0, compression_ratio=0.0, no_speech_prob=0.0) for output in outputs]
        LOGGER.info("Finished audio batch files=%d total_chars=%d elapsed=%s", len(audio_paths), sum(len(result.text) for result in results), _elapsed(start_time))
        return results


def _load_existing(output_csv: Path) -> pd.DataFrame:
    if output_csv.is_file():
        LOGGER.info("Loading existing predictions from %s", output_csv)
        df = pd.read_csv(output_csv, keep_default_na=False)
        LOGGER.info("Loaded existing predictions shape=%s", df.shape)
        return df
    LOGGER.info("No existing predictions found at %s", output_csv)
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def _write_log(log_jsonl: Path, row: dict) -> None:
    log_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with log_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    LOGGER.info("Appended JSONL log event to %s", log_jsonl)


def run_inference(
    sample_df: pd.DataFrame,
    audio_dir: Path,
    output_csv: Path,
    log_jsonl: Path,
    backend: WhisperPipelineBackend,
    normalization_policy: str,
    resume: bool,
    file_batch_size: int,
) -> pd.DataFrame:
    LOGGER.info("Starting inference")
    LOGGER.info("audio_dir=%s output_csv=%s log_jsonl=%s normalization_policy=%s resume=%s file_batch_size=%s", audio_dir, output_csv, log_jsonl, normalization_policy, resume, file_batch_size)
    start_time = time.perf_counter()
    existing = _load_existing(output_csv) if resume else pd.DataFrame(columns=PREDICTION_COLUMNS)
    completed = set(existing["file_name"].tolist()) if not existing.empty else set()
    rows = existing.to_dict("records")
    total = len(sample_df)
    LOGGER.info("Inference queue total=%d completed_from_resume=%d remaining=%d", total, len(completed), total - len(completed))

    pending = [
        (position, file_name, audio_dir / file_name)
        for position, file_name in enumerate(sample_df["file_name"].tolist(), start=1)
        if file_name not in completed
    ]
    if completed:
        LOGGER.info("Skipping %d completed files from resume", len(completed))

    for batch_start in range(0, len(pending), file_batch_size):
        batch = pending[batch_start : batch_start + file_batch_size]
        LOGGER.info("Processing file batch %d-%d of %d", batch_start + 1, batch_start + len(batch), len(pending))
        batch_start_time = time.perf_counter()
        try:
            batch_results = backend.transcribe_many([audio_path for _, _, audio_path in batch])
            if len(batch_results) != len(batch):
                raise RuntimeError(f"Expected {len(batch)} batched ASR outputs, got {len(batch_results)}")
        except Exception:
            LOGGER.exception("Batched inference failed; falling back to one-by-one for this batch")
            batch_results = []
            for _, file_name, audio_path in batch:
                try:
                    batch_results.append(backend.transcribe(audio_path))
                except Exception as exc:
                    LOGGER.exception("Inference failed for %s", file_name)
                    batch_results.append(exc)

        for (position, file_name, _), result in zip(batch, batch_results):
            file_start_time = time.perf_counter()
            if isinstance(result, Exception):
                row = {
                    "event": "prediction_error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "file_name": file_name,
                    "raw_text": "",
                    "normalized_text": "",
                    "model_name": backend.model_name,
                    "avg_logprob": 0.0,
                    "compression_ratio": 0.0,
                    "no_speech_prob": 0.0,
                    "error": f"{type(result).__name__}: {result}",
                    "traceback": "".join(traceback.format_exception(type(result), result, result.__traceback__)),
                    "elapsed_seconds": round(time.perf_counter() - file_start_time, 3),
                }
            else:
                raw_text = result.text
                normalized_text = normalize_text(raw_text, normalization_policy)
                row = {
                    "event": "prediction",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "file_name": file_name,
                    "raw_text": raw_text,
                    "normalized_text": normalized_text,
                    "model_name": backend.model_name,
                    "avg_logprob": result.avg_logprob,
                    "compression_ratio": result.compression_ratio,
                    "no_speech_prob": result.no_speech_prob,
                    "error": "",
                    "elapsed_seconds": round(time.perf_counter() - file_start_time, 3),
                }
                LOGGER.info("Prediction complete %d/%d %s raw_chars=%d normalized_chars=%d", position, total, file_name, len(raw_text), len(normalized_text))
            _write_log(log_jsonl, row)
            rows.append({column: row.get(column, "") for column in PREDICTION_COLUMNS})
        pd.DataFrame(rows, columns=PREDICTION_COLUMNS).to_csv(output_csv, index=False)
        LOGGER.info("Checkpointed predictions after %d/%d files to %s batch_elapsed=%s", len(rows), total, output_csv, _elapsed(batch_start_time))

    output = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    order = {file_name: position for position, file_name in enumerate(sample_df["file_name"].tolist())}
    output["_order"] = output["file_name"].map(order)
    output = output.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    output.to_csv(output_csv, index=False)
    error_count = int((output["error"] != "").sum()) if "error" in output else 0
    LOGGER.info("Inference finished rows=%d errors=%d elapsed=%s output_csv=%s", len(output), error_count, _elapsed(start_time), output_csv)
    return output


def require_cuda_runtime() -> None:
    import torch

    LOGGER.info("Checking CUDA runtime")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. In Colab, select Runtime > Change runtime type > A100 GPU.")
    LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(0))
    LOGGER.info("CUDA memory allocated=%d reserved=%d", torch.cuda.memory_allocated(0), torch.cuda.memory_reserved(0))

# %%
MODEL_NAME = "typhoon-ai/typhoon-whisper-large-v3"
NORMALIZATION_POLICY = "single_space"
CHUNK_LENGTH_SECONDS = 30
BATCH_SIZE = 32
FILE_BATCH_SIZE = 16
CONDITION_ON_PREVIOUS_TEXT = False
LOGGER.info("Notebook configuration set")
LOGGER.info("MODEL_NAME=%s", MODEL_NAME)
LOGGER.info("NORMALIZATION_POLICY=%s", NORMALIZATION_POLICY)
LOGGER.info("CHUNK_LENGTH_SECONDS=%s", CHUNK_LENGTH_SECONDS)
LOGGER.info("BATCH_SIZE=%s", BATCH_SIZE)
LOGGER.info("FILE_BATCH_SIZE=%s", FILE_BATCH_SIZE)
LOGGER.info("CONDITION_ON_PREVIOUS_TEXT=%s", CONDITION_ON_PREVIOUS_TEXT)

# %%
LOGGER.info("Data preparation cell started")
download_and_extract_competition_data()
paths = resolve_competition_paths(colab_input_root=CONTENT_ROOT / "input")
sample_df = load_sample_submission(paths.sample_submission)
validate_audio_coverage(sample_df, paths.audio_dir)
LOGGER.info("Rows in sample submission: %d", len(sample_df))
LOGGER.info("Audio directory: %s", paths.audio_dir)
LOGGER.info("Output submission path: %s", SUBMISSION_PATH)
LOGGER.info("Data preparation cell finished")

# %%
LOGGER.info("Model initialization cell started")
require_cuda_runtime()
backend = WhisperPipelineBackend(
    model_name=MODEL_NAME,
    chunk_length_s=CHUNK_LENGTH_SECONDS,
    batch_size=BATCH_SIZE,
    condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
    device="cuda",
)
LOGGER.info("Model initialization cell finished")

# %%
LOGGER.info("Inference cell started")
safe_model_name = MODEL_NAME.replace("/", "__")
predictions_path = paths.working_dir / f"predictions_{safe_model_name}_{NORMALIZATION_POLICY}.csv"
log_path = paths.working_dir / f"run_{safe_model_name}_{NORMALIZATION_POLICY}.jsonl"
LOGGER.info("Predictions path: %s", predictions_path)
LOGGER.info("JSONL run log path: %s", log_path)
predictions_df = run_inference(
    sample_df=sample_df,
    audio_dir=paths.audio_dir,
    output_csv=predictions_path,
    log_jsonl=log_path,
    backend=backend,
    normalization_policy=NORMALIZATION_POLICY,
    resume=True,
    file_batch_size=FILE_BATCH_SIZE,
)
LOGGER.info("Wrote predictions: %s", predictions_path)
LOGGER.info("Wrote logs: %s", log_path)
LOGGER.info("Prediction rows: %d", len(predictions_df))
LOGGER.info("Inference errors: %d", (predictions_df["error"] != "").sum())
LOGGER.info("Inference cell finished")

# %%
LOGGER.info("Submission writing cell started")
submission_df = write_submission_csv(
    sample_df=sample_df,
    predictions_df=predictions_df,
    output_path=SUBMISSION_PATH,
    allow_empty_files=set(),
)
validate_submission_frame(submission_df, allow_empty_files=set())
LOGGER.info("Wrote validated submission: %s", SUBMISSION_PATH)
LOGGER.info("Submission preview:\n%s", submission_df.head().to_string(index=False))
LOGGER.info("Submission writing cell finished")
